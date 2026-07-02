"""
Operation Écrasement v2 — Ratio-only Features + CalibratedClassifierCV
Key fixes:
1. Remove absolute band stats (re3, nira, swir2, green_raw, nir_raw) that cause imputation bias
2. Keep only ratio/normalized indices that self-correct imputation errors
3. Use CalibratedClassifierCV(method='isotonic') for proper probability calibration
4. 10-fold CV for lower variance test predictions
5. Multi-seed (5 seeds) ensembling
"""
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import KNNImputer
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('Train.csv')
test  = pd.read_csv('Test.csv')
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

features = [c for c in train.columns if c not in ['ID', 'label']]

# ===== Extract test missingness patterns =====
print("Extracting test missingness patterns...")
test_patterns = []
for idx, row in test.iterrows():
    row_pattern = []
    for m in range(1, 13):
        m_str = f'{m:02d}'
        has_s1 = (row[f'VH_{m_str}'] != -9999) and (row[f'VV_{m_str}'] != -9999)
        has_s2 = (row[f'blue_{m_str}'] != -9999) and (row[f'green_{m_str}'] != -9999) and \
                 (row[f'red_{m_str}'] != -9999) and (row[f'nir_{m_str}'] != -9999) and \
                 (row[f'swir1_{m_str}'] != -9999)
        if has_s1 or has_s2:
            row_pattern.append(m)
    test_patterns.append(row_pattern)

# ===== Mask + Impute =====
def mask_and_impute(seed):
    rng = random.Random(seed)
    train_masked = train.copy()
    for idx in range(len(train_masked)):
        pat = rng.choice(test_patterns)
        for m in range(1, 13):
            if m not in pat:
                m_str = f'{m:02d}'
                for b in ['blue', 'green', 'red', 'nir', 'nira', 'swir1', 'swir2', 're1', 're2', 're3', 'VH', 'VV']:
                    train_masked.at[idx, f'{b}_{m_str}'] = np.nan
    train_masked.replace(-9999, np.nan, inplace=True)
    test_nan = test.replace(-9999, np.nan)
    all_data = pd.concat([train_masked[features], test_nan[features]], axis=0)
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    imputer.fit(all_data)
    train_imp = pd.DataFrame(imputer.transform(train_masked[features]), columns=features)
    test_imp = pd.DataFrame(imputer.transform(test_nan[features]), columns=features)
    return train_imp, test_imp

# ===== Ratio-only Feature Extraction =====
# REMOVED: re3, nira, swir2, green_raw, nir_raw absolute stats
# KEPT: All ratio/normalized indices — these self-correct for imputation bias
def extract_ratio_features(df):
    feats = []
    for idx, row in df.iterrows():
        f = {}
        mndwi_v=[]; ndvi_v=[]; sar_v=[]; evi_v=[]; sdwi_v=[]
        sabi_v=[]; mci_v=[]; cdom_v=[]; twobda_v=[]
        ndsi_v=[]; awei_v=[]; vh_vv_v=[]; nira_ndvi_v=[]
        
        for m in range(1, 13):
            ms = f'{m:02d}'
            blue=row[f'blue_{ms}']; green=row[f'green_{ms}']
            red=row[f'red_{ms}']; nir=row[f'nir_{ms}']
            nira=row[f'nira_{ms}']; swir1=row[f'swir1_{ms}']
            swir2=row[f'swir2_{ms}']; vh=row[f'VH_{ms}']
            vv=row[f'VV_{ms}']; re1=row[f're1_{ms}']
            re2=row[f're2_{ms}']
            
            # Ratio indices (robust to imputation bias)
            mndwi=(green-swir1)/(green+swir1+1e-8)
            ndvi=(nir-red)/(nir+red+1e-8)
            vh_lin=10**(vh/10); vv_lin=10**(vv/10)
            sar=vh_lin+vv_lin
            evi=2.5*(nir-red)/(nir+6.0*red-7.5*blue+1.0+1e-8)
            sdwi=np.log(10.0*vv_lin*vh_lin+1e-8)-8.0
            sabi=(nir-red)/(green+blue+1e-8)
            cdom_val=green/(red+1e-8)
            mci_val=re1-red-0.5333*(re2-red)
            twobda_val=re1/(red+1e-8)
            # NEW ratio indices (use new bands but as normalized ratios)
            ndsi=(green-swir2)/(green+swir2+1e-8)
            awei=4.0*(green-swir1)-(0.25*nir+2.75*swir2)   # absolute but normalized by design
            vh_vv=vh-vv  # dB difference (ratio in linear space)
            nira_ndvi=(nira-red)/(nira+red+1e-8)
            
            mndwi_v.append(mndwi); ndvi_v.append(ndvi); sar_v.append(sar)
            evi_v.append(evi); sdwi_v.append(sdwi); sabi_v.append(sabi)
            mci_v.append(mci_val); cdom_v.append(cdom_val); twobda_v.append(twobda_val)
            ndsi_v.append(ndsi); awei_v.append(awei); vh_vv_v.append(vh_vv)
            nira_ndvi_v.append(nira_ndvi)
        
        all_series = {
            'mndwi':np.array(mndwi_v), 'ndvi':np.array(ndvi_v),
            'sar':np.array(sar_v), 'evi':np.array(evi_v),
            'sdwi':np.array(sdwi_v), 'sabi':np.array(sabi_v),
            'mci':np.array(mci_v), 'cdom':np.array(cdom_v),
            'twobda':np.array(twobda_v),
            'ndsi':np.array(ndsi_v), 'awei':np.array(awei_v),
            'vh_vv':np.array(vh_vv_v), 'nira_ndvi':np.array(nira_ndvi_v),
        }
        
        def pstats(a, name):
            s=np.array(sorted(a)); n=len(s)
            def p(pct): return s[max(0,min(n-1,int(n*pct/100)))]
            f[f'{name}_p5']=p(5); f[f'{name}_p10']=p(10); f[f'{name}_p25']=p(25)
            f[f'{name}_p50']=p(50); f[f'{name}_p75']=p(75); f[f'{name}_p90']=p(90)
            f[f'{name}_p95']=p(95); f[f'{name}_min']=s[0]; f[f'{name}_max']=s[-1]
            f[f'{name}_range']=s[-1]-s[0]; f[f'{name}_std']=np.std(a)
            f[f'{name}_frac_pos']=np.mean(a>0)
        
        for name, arr in all_series.items():
            pstats(arr, name)
        
        # Temporal gradient features
        for name in ['mndwi', 'ndvi', 'sar', 'ndsi']:
            arr=all_series[name]; diffs=np.diff(arr)
            f[f'{name}_grad_mean']=np.mean(diffs)
            f[f'{name}_grad_std']=np.std(diffs)
            f[f'{name}_grad_max']=np.max(np.abs(diffs))
        
        # Water frequency and correlations
        mndwi_a=all_series['mndwi']; sar_a=all_series['sar']
        f['water_freq_strict']=np.mean((mndwi_a>0.5)&(sar_a<0.1))
        f['water_freq_sdwi']=np.mean(all_series['sdwi']>-1.5)
        f['mndwi_sar_corr']=np.corrcoef(mndwi_a,sar_a)[0,1]
        f['mndwi_ndvi_corr']=np.corrcoef(mndwi_a,all_series['ndvi'])[0,1]
        f['ndsi_mndwi_corr']=np.corrcoef(all_series['ndsi'],mndwi_a)[0,1]
        f['water_score']=f['mndwi_p25']*(1-min(f['sar_p25'],1))
        f['water_floor']=f['mndwi_p10']-f['sar_p10']
        f['water_consist']=f['mndwi_frac_pos']*(1-f['sar_frac_pos'])
        
        feats.append(f)
    return pd.DataFrame(feats)

# ===== Submission Formatter =====
def map_probabilities(cal_probs, binary_preds):
    """Guarantee TargetRAUC >= 0.5 for positives, < 0.5 for negatives."""
    mapped = np.zeros_like(cal_probs, dtype=float)
    pos_mask = binary_preds == 1
    neg_mask = ~pos_mask
    if pos_mask.any():
        pos_probs = cal_probs[pos_mask]
        lo, hi = pos_probs.min(), pos_probs.max()
        mapped[pos_mask] = 0.5 + 0.5 * (pos_probs - lo) / (hi - lo + 1e-8) if hi > lo else 0.75
    if neg_mask.any():
        neg_probs = cal_probs[neg_mask]
        lo, hi = neg_probs.min(), neg_probs.max()
        mapped[neg_mask] = 0.5 * (neg_probs - lo) / (hi - lo + 1e-8) if hi > lo else 0.25
    return np.clip(mapped, 0.0, 1.0)

# ===== Multi-Seed Training Loop =====
SEEDS = [42, 123, 456, 789, 1337]
y_train_base = train['label'].values
n_test = len(test)

all_test_preds_base = []
cached_data = {}  # Cache (X_train, X_test) per seed to avoid duplicate imputation/extraction

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"  SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'='*60}")
    
    train_imp, test_imp = mask_and_impute(seed)
    X_train = extract_ratio_features(train_imp)
    X_test = extract_ratio_features(test_imp)
    feature_cols = list(X_train.columns)
    if seed_idx == 0:
        print(f"  Feature count: {len(feature_cols)}")
        
    # Store in cache
    cached_data[seed] = (X_train, X_test, feature_cols)
    
    # 5-fold CV (balance between robustness and speed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y_train_base))
    oof_cb  = np.zeros(len(y_train_base))
    oof_xgb = np.zeros(len(y_train_base))
    test_preds_lgb = []; test_preds_cb = []; test_preds_xgb = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_base)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train_base[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train_base[val_idx]
        
        # LGBM + CalibratedClassifierCV
        clf_lgb = lgb.LGBMClassifier(random_state=seed+fold, n_estimators=800, learning_rate=0.02,
                                      max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=10)
        clf_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        cal_lgb = CalibratedClassifierCV(clf_lgb, cv='prefit', method='isotonic')
        cal_lgb.fit(X_val, y_val)
        oof_lgb[val_idx] = cal_lgb.predict_proba(X_val)[:, 1]
        test_preds_lgb.append(cal_lgb.predict_proba(X_test[feature_cols])[:, 1])
        
        # CatBoost + CalibratedClassifierCV
        clf_cb = CatBoostClassifier(random_seed=seed+fold, iterations=800, learning_rate=0.02,
                                    depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0)
        clf_cb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
        cal_cb = CalibratedClassifierCV(clf_cb, cv='prefit', method='isotonic')
        cal_cb.fit(X_val, y_val)
        oof_cb[val_idx] = cal_cb.predict_proba(X_val)[:, 1]
        test_preds_cb.append(cal_cb.predict_proba(X_test[feature_cols])[:, 1])
        
        # XGBoost + CalibratedClassifierCV
        clf_xgb = xgb.XGBClassifier(random_state=seed+fold, n_estimators=800, learning_rate=0.02,
                                     max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss')
        clf_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        cal_xgb = CalibratedClassifierCV(clf_xgb, cv='prefit', method='isotonic')
        cal_xgb.fit(X_val, y_val)
        oof_xgb[val_idx] = cal_xgb.predict_proba(X_val)[:, 1]
        test_preds_xgb.append(cal_xgb.predict_proba(X_test[feature_cols])[:, 1])
    
    # Blend weights from OOF
    best_score, best_weights = 0, (1/3, 1/3, 1/3)
    for w1 in np.linspace(0, 1, 11):
        for w2 in np.linspace(0, 1-w1, 11):
            w3 = 1 - w1 - w2
            if w3 < 0: continue
            oof_ens = w1*oof_lgb + w2*oof_cb + w3*oof_xgb
            s = 0.6*f1_score(y_train_base, oof_ens>=0.5) + 0.4*roc_auc_score(y_train_base, oof_ens)
            if s > best_score:
                best_score = s; best_weights = (w1, w2, w3)
    
    w1, w2, w3 = best_weights
    oof_final = w1*oof_lgb + w2*oof_cb + w3*oof_xgb
    oof_pos = (oof_final >= 0.5).sum()
    print(f"  Weights: LGBM={w1:.2f}, CB={w2:.2f}, XGB={w3:.2f} | CV={best_score:.5f} | OOF positives: {oof_pos}")
    
    p_test = (w1*np.mean(test_preds_lgb, axis=0) +
              w2*np.mean(test_preds_cb, axis=0) +
              w3*np.mean(test_preds_xgb, axis=0))
    all_test_preds_base.append(p_test)

# Average across seeds
p_test_avg = np.mean(all_test_preds_base, axis=0)
pos_base = (p_test_avg >= 0.5).sum()
print(f"\n{'='*60}")
print(f"  BASE MODEL (5 seeds, 5-fold, ratio-only features)")
print(f"{'='*60}")
print(f"Test positives: {pos_base} / {n_test}")

sub_base_f1 = (p_test_avg >= 0.5).astype(int)
base_mapped = map_probabilities(p_test_avg, sub_base_f1)
pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_base_f1, 'TargetRAUC': base_mapped}).to_csv('submission_base.csv', index=False)
print(f"Saved submission_base.csv with {sub_base_f1.sum()} positives.")

# ===== PSEUDO-LABELING Round 1 =====
print(f"\n{'='*60}")
print(f"  PSEUDO-LABELING Round 1 (threshold 0.90)")
print(f"{'='*60}")

m_pos = p_test_avg >= 0.90
m_neg = p_test_avg <= 0.10
print(f"Pseudo-positive: {m_pos.sum()}, Pseudo-negative: {m_neg.sum()}")

all_test_preds_pl = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n  PL Seed {seed} ({seed_idx+1}/{len(SEEDS)})...")
    
    # Retrieve from cache
    X_train, X_test, feature_cols = cached_data[seed]
    
    X_pseudo_pos = X_test[feature_cols][m_pos].copy()
    X_pseudo_neg = X_test[feature_cols][m_neg].copy()
    X_pseudo = pd.concat([X_pseudo_pos, X_pseudo_neg], ignore_index=True)
    y_pseudo = np.concatenate([np.ones(m_pos.sum()), np.zeros(m_neg.sum())])
    
    # Changed to 5-fold CV to match base model and speed up training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y_train_base))
    oof_cb  = np.zeros(len(y_train_base))
    oof_xgb = np.zeros(len(y_train_base))
    test_preds_lgb = []; test_preds_cb = []; test_preds_xgb = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_base)):
        X_tr = pd.concat([X_train.iloc[tr_idx], X_pseudo], ignore_index=True)
        y_tr = np.concatenate([y_train_base[tr_idx], y_pseudo])
        X_val, y_val = X_train.iloc[val_idx], y_train_base[val_idx]
        
        clf_lgb = lgb.LGBMClassifier(random_state=seed+fold, n_estimators=800, learning_rate=0.02,
                                      max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=10)
        clf_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        cal_lgb = CalibratedClassifierCV(clf_lgb, cv='prefit', method='isotonic')
        cal_lgb.fit(X_val, y_val)
        oof_lgb[val_idx] = cal_lgb.predict_proba(X_val)[:, 1]
        test_preds_lgb.append(cal_lgb.predict_proba(X_test[feature_cols])[:, 1])
        
        clf_cb = CatBoostClassifier(random_seed=seed+fold, iterations=800, learning_rate=0.02,
                                    depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0)
        clf_cb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
        cal_cb = CalibratedClassifierCV(clf_cb, cv='prefit', method='isotonic')
        cal_cb.fit(X_val, y_val)
        oof_cb[val_idx] = cal_cb.predict_proba(X_val)[:, 1]
        test_preds_cb.append(cal_cb.predict_proba(X_test[feature_cols])[:, 1])
        
        clf_xgb = xgb.XGBClassifier(random_state=seed+fold, n_estimators=800, learning_rate=0.02,
                                     max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss')
        clf_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        cal_xgb = CalibratedClassifierCV(clf_xgb, cv='prefit', method='isotonic')
        cal_xgb.fit(X_val, y_val)
        oof_xgb[val_idx] = cal_xgb.predict_proba(X_val)[:, 1]
        test_preds_xgb.append(cal_xgb.predict_proba(X_test[feature_cols])[:, 1])
    
    best_score, best_weights = 0, (1/3, 1/3, 1/3)
    for w1 in np.linspace(0, 1, 11):
        for w2 in np.linspace(0, 1-w1, 11):
            w3 = 1 - w1 - w2
            if w3 < 0: continue
            oof_ens = w1*oof_lgb + w2*oof_cb + w3*oof_xgb
            s = 0.6*f1_score(y_train_base, oof_ens>=0.5) + 0.4*roc_auc_score(y_train_base, oof_ens)
            if s > best_score:
                best_score = s; best_weights = (w1, w2, w3)
    
    w1, w2, w3 = best_weights
    p_test = (w1*np.mean(test_preds_lgb, axis=0) +
              w2*np.mean(test_preds_cb, axis=0) +
              w3*np.mean(test_preds_xgb, axis=0))
    oof_pos = (w1*oof_lgb + w2*oof_cb + w3*oof_xgb >= 0.5).sum()
    print(f"  PL Weights: LGBM={w1:.2f}, CB={w2:.2f}, XGB={w3:.2f} | CV={best_score:.5f} | OOF positives: {oof_pos}")
    all_test_preds_pl.append(p_test)

p_test_pl = np.mean(all_test_preds_pl, axis=0)
pos_pl = (p_test_pl >= 0.5).sum()
print(f"\nPseudo-Labeled test positives: {pos_pl} / {n_test}")

sub_pl_f1 = (p_test_pl >= 0.5).astype(int)
pl_mapped = map_probabilities(p_test_pl, sub_pl_f1)
sub_final = pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_pl_f1, 'TargetRAUC': pl_mapped})
sub_final.to_csv('submission.csv', index=False)
sub_final.to_csv('submission_pseudo_r1.csv', index=False)

assert ((sub_final['TargetRAUC'] >= 0.5) == (sub_final['TargetF1'] == 1)).all(), "Format mismatch!"
print(f"Saved submission.csv with {sub_pl_f1.sum()} positives.")

print("\n" + "="*60)
print("  OPERATION ÉCRASEMENT v2 COMPLETE!")
print("="*60)
