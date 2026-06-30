"""
Final Improved Pipeline for Aquaculture Pond Identification Challenge

Key Enhancements implemented:
1. Split S1/S2 Feature Extraction: Extracts S1 (radar) and S2 (optical) features
   independently for each month. Discarding optical data due to cloud cover
   no longer results in throwing away valuable radar observations.
2. New Physical Features: Statistics of raw VH and VV bands (dB and linear scales)
   and temporal correlation between MNDWI and NDVI.
3. Clean Transductive Self-Training: Harvesting high-confidence pseudo-labels
   from the test set (prob >= 0.90 or <= 0.10) and appending them cleanly to
   the training splits without duplicating/triplicating across iterations.
4. Robust 5-Fold Cross-Validation Ensemble: Training LightGBM, CatBoost, and XGBoost
   in a 5-fold CV scheme, averaging predictions over all folds, and calibrating
   probabilities using Isotonic Regression fitted on clean OOF predictions.
"""
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

print("Loading data...")
train = pd.read_csv('Train.csv')
test  = pd.read_csv('Test.csv')

print(f"Train set: {train.shape}, Test set: {test.shape}")

# Extract S1/S2 availability patterns from test set
test_patterns = []
for idx, row in test.iterrows():
    row_pattern = []
    for m in range(1, 13):
        m_str = f'{m:02d}'
        has_s1 = (row[f'VH_{m_str}'] != -9999) and (row[f'VV_{m_str}'] != -9999)
        has_s2 = (row[f'blue_{m_str}'] != -9999) and (row[f'green_{m_str}'] != -9999) and \
                 (row[f'red_{m_str}'] != -9999) and (row[f'nir_{m_str}'] != -9999) and \
                 (row[f'swir1_{m_str}'] != -9999)
        
        if has_s1 and has_s2:
            status = "both"
        elif has_s1 and not has_s2:
            status = "s1_only"
        elif not has_s1 and has_s2:
            status = "s2_only"
        else:
            status = "none"
        row_pattern.append((m, status))
    obs_pattern = [p for p in row_pattern if p[1] != "none"]
    test_patterns.append(obs_pattern)

def compute_monthly_all(row, month, sensor_status):
    ms = f'{month:02d}'
    blue=row[f'blue_{ms}']; green=row[f'green_{ms}']; red=row[f'red_{ms}']
    nir=row[f'nir_{ms}']; swir1=row[f'swir1_{ms}']; swir2=row[f'swir2_{ms}']
    vh=row[f'VH_{ms}']; vv=row[f'VV_{ms}']
    re1=row[f're1_{ms}']; re2=row[f're2_{ms}']
    
    res = {}
    if sensor_status in ['both', 's1_only']:
        if vh != -9999 and vv != -9999:
            vh_lin = 10**(vh/10)
            vv_lin = 10**(vv/10)
            sar = vh_lin + vv_lin
            res.update({
                'vh': vh,
                'vv': vv,
                'vh_lin': vh_lin,
                'vv_lin': vv_lin,
                'sar': sar,
                'rvi': 4*vh_lin/(sar+1e-8),
                'sar_diff': vv - vh
            })
            
    if sensor_status in ['both', 's2_only']:
        if blue != -9999 and green != -9999 and red != -9999 and nir != -9999 and swir1 != -9999:
            ndre = (re2-re1)/(re2+re1+1e-8) if re1!=-9999 and re2!=-9999 else np.nan
            ndci = (re1-red)/(re1+red+1e-8) if re1!=-9999 else np.nan
            res.update({
                'mndwi': (green-swir1)/(green+swir1+1e-8),
                'ndwi': (green-nir)/(green+nir+1e-8),
                'ndvi': (nir-red)/(nir+red+1e-8),
                'lswi': (nir-swir1)/(nir+swir1+1e-8),
                'awei': 4*(green-swir1)-(0.25*nir+2.75*swir2),
                'ndre': ndre,
                'ndci': ndci,
                'bsi': ((swir1+red)-(nir+blue))/((swir1+red)+(nir+blue)+1e-8)
            })
            
    return res

def extract_features_split(row, pattern_obs, add_new_feats=True):
    feats = {}
    
    obs_s1 = [m for m, status in pattern_obs if status in ['both', 's1_only']]
    obs_s2 = [m for m, status in pattern_obs if status in ['both', 's2_only']]
    
    feats['num_observed_s1'] = len(obs_s1)
    feats['num_observed_s2'] = len(obs_s2)
    
    if obs_s1:
        feats['window_size_s1'] = max(obs_s1)-min(obs_s1)+1
        feats['start_month_s1'] = min(obs_s1)
        feats['end_month_s1']   = max(obs_s1)
    else:
        feats['window_size_s1'] = feats['start_month_s1'] = feats['end_month_s1'] = 0
        
    if obs_s2:
        feats['window_size_s2'] = max(obs_s2)-min(obs_s2)+1
        feats['start_month_s2'] = min(obs_s2)
        feats['end_month_s2']   = max(obs_s2)
    else:
        feats['window_size_s2'] = feats['start_month_s2'] = feats['end_month_s2'] = 0

    mv = {}
    for m, status in pattern_obs:
        v = compute_monthly_all(row, m, status)
        if v:
            mv[m] = v

    def arr(key):
        return np.array([mv[m][key] for m in mv if key in mv[m] and not np.isnan(mv[m][key])])

    def pstats(a, name):
        if len(a)==0:
            for s in ['p5','p10','p25','p50','p75','p90','p95','min','max','range','std','frac_pos']:
                feats[f'{name}_{s}']=np.nan
            return
        s=np.array(sorted(a)); n=len(s)
        def p(pct): return s[max(0,min(n-1,int(n*pct/100)))]
        feats[f'{name}_p5']=p(5); feats[f'{name}_p10']=p(10); feats[f'{name}_p25']=p(25)
        feats[f'{name}_p50']=p(50); feats[f'{name}_p75']=p(75); feats[f'{name}_p90']=p(90)
        feats[f'{name}_p95']=p(95); feats[f'{name}_min']=s[0]; feats[f'{name}_max']=s[-1]
        feats[f'{name}_range']=s[-1]-s[0]; feats[f'{name}_std']=np.std(a)
        feats[f'{name}_frac_pos']=np.mean(a>0)

    keys = ['mndwi','ndwi','ndvi','lswi','awei','sar','rvi','sar_diff','ndre','ndci','bsi']
    if add_new_feats:
        keys += ['vh', 'vv', 'vh_lin', 'vv_lin']
        
    for key in keys:
        pstats(arr(key), key)

    mndwi_a = arr('mndwi')
    sar_a = arr('sar')
    if len(mndwi_a) > 0 and len(sar_a) > 0:
        feats['water_score'] = feats['mndwi_p25'] * (1 - min(feats['sar_p25'], 1))
        feats['water_floor'] = feats['mndwi_p10'] - feats['sar_p10']
        feats['water_consist'] = feats['mndwi_frac_pos'] * (1 - feats['sar_frac_pos'])
        feats['min_mndwi_pos'] = float(feats['mndwi_min'] > 0)
        feats['max_sar_low'] = float(feats['sar_max'] < 0.1)
        
        both_ms = [m for m, status in pattern_obs if status == 'both']
        if len(both_ms) >= 2:
            mvals = np.array([mv[m]['mndwi'] for m in both_ms if 'mndwi' in mv[m] and 'sar' in mv[m]])
            svals = np.array([mv[m]['sar'] for m in both_ms if 'mndwi' in mv[m] and 'sar' in mv[m]])
            if len(mvals) >= 2 and np.std(mvals)>0 and np.std(svals)>0:
                feats['mndwi_sar_corr'] = np.corrcoef(mvals, svals)[0, 1]
            else:
                feats['mndwi_sar_corr'] = 0
                
            if add_new_feats:
                ndvi_vals = np.array([mv[m]['ndvi'] for m in both_ms if 'mndwi' in mv[m] and 'ndvi' in mv[m]])
                if len(mvals) >= 2 and np.std(mvals)>0 and np.std(ndvi_vals)>0:
                    feats['mndwi_ndvi_corr'] = np.corrcoef(mvals, ndvi_vals)[0, 1]
                else:
                    feats['mndwi_ndvi_corr'] = 0
        else:
            feats['mndwi_sar_corr'] = 0
            if add_new_feats:
                feats['mndwi_ndvi_corr'] = 0
    else:
        for k in ['water_score','water_floor','water_consist','min_mndwi_pos','max_sar_low','mndwi_sar_corr']:
            feats[k] = np.nan
        if add_new_feats:
            feats['mndwi_ndvi_corr'] = np.nan
            
    return feats

print("\nExtracting test features...")
test_rows = []
for idx, row in test.iterrows():
    pat = test_patterns[idx]
    test_rows.append(extract_features_split(row, pat, add_new_feats=True))
X_test = pd.DataFrame(test_rows)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Extracting train features...")
train_rows = []
for idx, row in train.iterrows():
    train_pattern = [(m, 'both') for m in range(1, 13)]
    train_rows.append(extract_features_split(row, train_pattern, add_new_feats=True))
X_train_base = pd.DataFrame(train_rows)
X_train_base.replace([np.inf, -np.inf], np.nan, inplace=True)
y_train_base = train['label'].values

feature_cols = [c for c in X_train_base.columns]
print(f"Number of features extracted: {len(feature_cols)}")

def train_ensemble_cv(X_tr_base, y_tr_base, X_pseudo=None, y_pseudo=None):
    """
    Trains a 5-fold ensemble of LGBM, CatBoost, and XGBoost.
    Pseudo-labels are added only to the training folds, keeping validation folds completely clean.
    Returns:
      - list of trained models (5 folds * 3 architectures = 15 models)
      - OOF probabilities for the baseline training samples
      - Averaged predictions on the test set
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y_tr_base))
    oof_cb  = np.zeros(len(y_tr_base))
    oof_xgb = np.zeros(len(y_tr_base))
    
    test_preds_lgb = []
    test_preds_cb  = []
    test_preds_xgb = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_base, y_tr_base)):
        print(f"  Training Fold {fold}...")
        
        # Split baseline train/val
        X_fold_tr, y_fold_tr = X_tr_base.iloc[tr_idx].copy(), y_tr_base[tr_idx]
        X_fold_val, y_fold_val = X_tr_base.iloc[val_idx].copy(), y_tr_base[val_idx]
        
        # If transductive pseudo-labels are provided, append them cleanly to the train fold
        if X_pseudo is not None and y_pseudo is not None:
            X_fold_tr = pd.concat([X_fold_tr, X_pseudo], ignore_index=True)
            y_fold_tr = np.concatenate([y_fold_tr, y_pseudo])
            
        # 1. Train LGBM
        clf_lgb = lgb.LGBMClassifier(random_state=42+fold, n_estimators=600, learning_rate=0.03,
                                      max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=10)
        clf_lgb.fit(X_fold_tr, y_fold_tr, eval_set=[(X_fold_val, y_fold_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = clf_lgb.predict_proba(X_fold_val)[:, 1]
        test_preds_lgb.append(clf_lgb.predict_proba(X_test[feature_cols])[:, 1])
        
        # 2. Train CatBoost
        clf_cb = CatBoostClassifier(random_seed=42+fold, iterations=600, learning_rate=0.03,
                                    depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0)
        clf_cb.fit(X_fold_tr, y_fold_tr, eval_set=[(X_fold_val, y_fold_val)], early_stopping_rounds=50)
        oof_cb[val_idx] = clf_cb.predict_proba(X_fold_val)[:, 1]
        test_preds_cb.append(clf_cb.predict_proba(X_test[feature_cols])[:, 1])
        
        # 3. Train XGBoost
        clf_xgb = xgb.XGBClassifier(random_state=42+fold, n_estimators=600, learning_rate=0.03,
                                     max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss')
        clf_xgb.fit(X_fold_tr, y_fold_tr, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
        oof_xgb[val_idx] = clf_xgb.predict_proba(X_fold_val)[:, 1]
        test_preds_xgb.append(clf_xgb.predict_proba(X_test[feature_cols])[:, 1])

    # Find best ensemble weights using OOF scores
    best_score, best_weights = 0, (1/3, 1/3, 1/3)
    for w1 in np.linspace(0, 1, 11):
        for w2 in np.linspace(0, 1-w1, 11):
            w3 = 1 - w1 - w2
            if w3 < 0: continue
            oof_ens = w1*oof_lgb + w2*oof_cb + w3*oof_xgb
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(oof_ens, y_tr_base)
            cal_oof = iso.predict(oof_ens)
            s = 0.6*f1_score(y_tr_base, cal_oof>=0.5) + 0.4*roc_auc_score(y_tr_base, cal_oof)
            if s > best_score:
                best_score = s
                best_weights = (w1, w2, w3)
                
    w1, w2, w3 = best_weights
    print(f"  Best fold weights: LGBM={w1:.2f}, CatBoost={w2:.2f}, XGBoost={w3:.2f} | CV Score: {best_score:.5f}")
    
    # Calculate calibrated test predictions
    oof_final = w1*oof_lgb + w2*oof_cb + w3*oof_xgb
    iso_final = IsotonicRegression(out_of_bounds='clip')
    iso_final.fit(oof_final, y_tr_base)
    
    p_test_avg = (w1 * np.mean(test_preds_lgb, axis=0) +
                  w2 * np.mean(test_preds_cb, axis=0) +
                  w3 * np.mean(test_preds_xgb, axis=0))
    p_test_cal = iso_final.predict(p_test_avg)
    
    return oof_final, p_test_cal

# ===== ITERATION 0: Base Model Training =====
print("\n===== Running Base Model (Iteration 0) =====")
oof_0, p_test_0 = train_ensemble_cv(X_train_base, y_train_base)
pos_0 = (p_test_0 >= 0.5).sum()
print(f"Base Model test set positives: {pos_0} / {len(p_test_0)}")

# ===== ITERATION 1: Transductive Self-Training =====
print("\n===== Harvesting High-Confidence Pseudo-Labels =====")
CONF_THRESHOLD = 0.90
mask_pos = p_test_0 >= CONF_THRESHOLD
mask_neg = p_test_0 <= (1 - CONF_THRESHOLD)

print(f"Pseudo-positive test cases: {mask_pos.sum()}")
print(f"Pseudo-negative test cases: {mask_neg.sum()}")

X_pseudo_pos = X_test[feature_cols][mask_pos].copy()
X_pseudo_pos['label'] = 1
X_pseudo_neg = X_test[feature_cols][mask_neg].copy()
X_pseudo_neg['label'] = 0

X_pseudo = pd.concat([X_pseudo_pos, X_pseudo_neg], ignore_index=True)
y_pseudo = X_pseudo['label'].values
X_pseudo = X_pseudo[feature_cols]

print(f"Total pseudo-labels added to training splits: {len(X_pseudo)}")

print("\n===== Running Self-Trained Model (Iteration 1) =====")
oof_1, p_test_1 = train_ensemble_cv(X_train_base, y_train_base, X_pseudo=X_pseudo, y_pseudo=y_pseudo)
pos_1 = (p_test_1 >= 0.5).sum()
print(f"Self-trained Model test set positives: {pos_1} / {len(p_test_1)}")

# ===== Generate Final Submission =====
print("\nGenerating final submission files...")
sub_final = pd.DataFrame({
    'ID': test['ID'],
    'TargetF1': (p_test_1 >= 0.5).astype(int),
    'TargetRAUC': p_test_1
})

# Save to submission.csv and also submission_improved_final.csv
sub_final.to_csv('submission.csv', index=False)
sub_final.to_csv('submission_improved_final.csv', index=False)

print("\nSubmissions saved successfully!")
print("Class distribution:\n", sub_final['TargetF1'].value_counts())
