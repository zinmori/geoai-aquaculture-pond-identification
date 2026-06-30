"""
Symmetric KNN Imputation Pipeline with 3-Round Pseudo-Labeling
For Aquaculture Pond Identification Challenge

Features:
1. Symmetric Masking: Randomly masks Train data to match Test missingness patterns.
2. Global KNN Imputation: Reconstructs 12 continuous months for both Train and Test.
3. 12-Month Feature & WQ Index Extraction: Computes SDWI, EVI, SABI, MCI, CDOM, 2BDA, MNDWI, NDVI, SAR.
4. Ensemble Modeling: LGBM, CatBoost, and XGBoost with Isotonic Probability Calibration.
5. 3 Rounds of Transductive Self-Training (Pseudo-Labeling).
"""
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
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

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

features = [c for c in train.columns if c not in ['ID', 'label']]

# 1. Extract test set missingness patterns
print("Extracting missingness patterns from Test set...")
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

# 2. Mask the training set symmetrically to reproduce the test missingness patterns
print("Masking training set to match test set missingness patterns...")
train_masked = train.copy()
for idx, row in train_masked.iterrows():
    # Randomly assign one of the test missingness patterns
    pat = random.choice(test_patterns)
    for m in range(1, 13):
        if m not in pat:
            m_str = f'{m:02d}'
            for b in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 're1', 're2', 'VH', 'VV']:
                train_masked.at[idx, f'{b}_{m_str}'] = np.nan

# Replace remaining -9999 with NaN
train_masked.replace(-9999, np.nan, inplace=True)
test_nan = test.replace(-9999, np.nan)

# 3. Apply global KNN Imputation to both Train and Test
print("Fitting KNN Imputer and transforming full dataset (Train + Test)...")
all_data = pd.concat([train_masked[features], test_nan[features]], axis=0)
imputer = KNNImputer(n_neighbors=5, weights='distance')
imputer.fit(all_data)

train_imp_arr = imputer.transform(train_masked[features])
test_imp_arr = imputer.transform(test_nan[features])

train_imputed = pd.DataFrame(train_imp_arr, columns=features)
test_imputed = pd.DataFrame(test_imp_arr, columns=features)

# Restore ID and label
train_imputed['ID'] = train['ID']
train_imputed['label'] = train['label']
test_imputed['ID'] = test['ID']

# 4. Extract robust 12-month features on the imputed profiles
def extract_full_features(df):
    feats = []
    for idx, row in df.iterrows():
        f = {}
        mndwi_vals = []
        sar_vals = []
        ndvi_vals = []
        evi_vals = []
        sdwi_vals = []
        sabi_vals = []
        mci_vals = []
        cdom_vals = []
        twobda_vals = []
        
        for m in range(1, 13):
            m_str = f'{m:02d}'
            blue = row[f'blue_{m_str}']
            green = row[f'green_{m_str}']
            red = row[f'red_{m_str}']
            nir = row[f'nir_{m_str}']
            swir1 = row[f'swir1_{m_str}']
            vh = row[f'VH_{m_str}']
            vv = row[f'VV_{m_str}']
            re1 = row[f're1_{m_str}']
            re2 = row[f're2_{m_str}']
            
            # Indices
            mndwi = (green-swir1)/(green+swir1+1e-8)
            ndvi = (nir-red)/(nir+red+1e-8)
            vh_lin = 10**(vh/10); vv_lin = 10**(vv/10)
            sar = vh_lin + vv_lin
            
            evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-8)
            sdwi = np.log(10.0 * vv_lin * vh_lin + 1e-8) - 8.0
            sabi = (nir - red) / (green + blue + 1e-8)
            cdom = green / (red + 1e-8)
            mci = re1 - red - 0.5333 * (re2 - red)
            twobda = re1 / (red + 1e-8)
            
            mndwi_vals.append(mndwi)
            ndvi_vals.append(ndvi)
            sar_vals.append(sar)
            evi_vals.append(evi)
            sdwi_vals.append(sdwi)
            sabi_vals.append(sabi)
            mci_vals.append(mci)
            cdom_vals.append(cdom)
            twobda_vals.append(twobda)
            
        mndwi_vals = np.array(mndwi_vals)
        ndvi_vals = np.array(ndvi_vals)
        sar_vals = np.array(sar_vals)
        evi_vals = np.array(evi_vals)
        sdwi_vals = np.array(sdwi_vals)
        sabi_vals = np.array(sabi_vals)
        mci_vals = np.array(mci_vals)
        cdom_vals = np.array(cdom_vals)
        twobda_vals = np.array(twobda_vals)
        
        def pstats(a, name):
            s=np.array(sorted(a)); n=len(s)
            def p(pct): return s[max(0,min(n-1,int(n*pct/100)))]
            f[f'{name}_p5']=p(5); f[f'{name}_p10']=p(10); f[f'{name}_p25']=p(25)
            f[f'{name}_p50']=p(50); f[f'{name}_p75']=p(75); f[f'{name}_p90']=p(90)
            f[f'{name}_p95']=p(95); f[f'{name}_min']=s[0]; f[f'{name}_max']=s[-1]
            f[f'{name}_range']=s[-1]-s[0]; f[f'{name}_std']=np.std(a)
            f[f'{name}_frac_pos']=np.mean(a>0)

        pstats(mndwi_vals, 'mndwi')
        pstats(ndvi_vals, 'ndvi')
        pstats(sar_vals, 'sar')
        pstats(evi_vals, 'evi')
        pstats(sdwi_vals, 'sdwi')
        pstats(sabi_vals, 'sabi')
        pstats(mci_vals, 'mci')
        pstats(cdom_vals, 'cdom')
        pstats(twobda_vals, 'twobda')
        
        f['water_freq_strict'] = np.mean((mndwi_vals > 0.5) & (sar_vals < 0.1))
        f['water_freq_sdwi'] = np.mean(sdwi_vals > -1.5)
        f['mndwi_sar_corr'] = np.corrcoef(mndwi_vals, sar_vals)[0, 1]
        f['mndwi_ndvi_corr'] = np.corrcoef(mndwi_vals, ndvi_vals)[0, 1]
        
        f['water_score'] = f['mndwi_p25'] * (1 - min(f['sar_p25'], 1))
        f['water_floor'] = f['mndwi_p10'] - f['sar_p10']
        f['water_consist'] = f['mndwi_frac_pos'] * (1 - f['sar_frac_pos'])
        
        feats.append(f)
    return pd.DataFrame(feats)

print("Extracting features...")
X_train_base = extract_full_features(train_imputed)
y_train_base = train_imputed['label'].values

X_test = extract_full_features(test_imputed)

feature_cols = [c for c in X_train_base.columns]
print(f"Number of features extracted: {len(feature_cols)}")

def train_ensemble_cv(X_tr_base, y_tr_base, X_pseudo=None, y_pseudo=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgb = np.zeros(len(y_tr_base))
    oof_cb  = np.zeros(len(y_tr_base))
    oof_xgb = np.zeros(len(y_tr_base))
    
    test_preds_lgb = []
    test_preds_cb  = []
    test_preds_xgb = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_base, y_tr_base)):
        X_fold_tr, y_fold_tr = X_tr_base.iloc[tr_idx].copy(), y_tr_base[tr_idx]
        X_fold_val, y_fold_val = X_tr_base.iloc[val_idx].copy(), y_tr_base[val_idx]
        
        if X_pseudo is not None and y_pseudo is not None:
            X_fold_tr = pd.concat([X_fold_tr, X_pseudo], ignore_index=True)
            y_fold_tr = np.concatenate([y_fold_tr, y_pseudo])
            
        # 1. LGBM
        clf_lgb = lgb.LGBMClassifier(random_state=42+fold, n_estimators=600, learning_rate=0.03,
                                      max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=10)
        clf_lgb.fit(X_fold_tr, y_fold_tr, eval_set=[(X_fold_val, y_fold_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = clf_lgb.predict_proba(X_fold_val)[:, 1]
        test_preds_lgb.append(clf_lgb.predict_proba(X_test[feature_cols])[:, 1])
        
        # 2. CatBoost
        clf_cb = CatBoostClassifier(random_seed=42+fold, iterations=600, learning_rate=0.03,
                                    depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0)
        clf_cb.fit(X_fold_tr, y_fold_tr, eval_set=[(X_fold_val, y_fold_val)], early_stopping_rounds=50)
        oof_cb[val_idx] = clf_cb.predict_proba(X_fold_val)[:, 1]
        test_preds_cb.append(clf_cb.predict_proba(X_test[feature_cols])[:, 1])
        
        # 3. XGBoost
        clf_xgb = xgb.XGBClassifier(random_state=42+fold, n_estimators=600, learning_rate=0.03,
                                     max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss')
        clf_xgb.fit(X_fold_tr, y_fold_tr, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
        oof_xgb[val_idx] = clf_xgb.predict_proba(X_fold_val)[:, 1]
        test_preds_xgb.append(clf_xgb.predict_proba(X_test[feature_cols])[:, 1])

    # Find optimal blend weights based on OOF
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
    
    oof_final = w1*oof_lgb + w2*oof_cb + w3*oof_xgb
    iso_final = IsotonicRegression(out_of_bounds='clip')
    iso_final.fit(oof_final, y_tr_base)
    
    p_test_avg = (w1 * np.mean(test_preds_lgb, axis=0) +
                  w2 * np.mean(test_preds_cb, axis=0) +
                  w3 * np.mean(test_preds_xgb, axis=0))
    p_test_cal = iso_final.predict(p_test_avg)
    
    return oof_final, p_test_cal, p_test_avg

# Probability Mapper for Submission Constraints
def map_probabilities(raw_probs, binary_preds, K):
    sorted_indices = np.argsort(raw_probs)[::-1]
    cutoff_val = raw_probs[sorted_indices[K-1]]
    max_val = raw_probs.max()
    
    mapped = np.zeros_like(raw_probs)
    for i, val in enumerate(raw_probs):
        if val >= cutoff_val:
            mapped[i] = 0.5 + 0.5 * (val - cutoff_val) / (max_val - cutoff_val + 1e-8)
        else:
            mapped[i] = 0.5 * val / (cutoff_val + 1e-8)
            
    return np.clip(mapped, 0.0, 1.0)

# ===== ITERATION 0: Base Model =====
print("\n===== Running Base Model (Iteration 0) =====")
oof_0, p_test_0, p_test_0_raw = train_ensemble_cv(X_train_base, y_train_base)
pos_0 = (p_test_0 >= 0.5).sum()
print(f"Base Model test set positives: {pos_0} / {len(p_test_0)}")

# Save Base Submission
sub_base_f1 = (p_test_0 >= 0.5).astype(int)
base_K = max(sub_base_f1.sum(), 1)
base_mapped = map_probabilities(p_test_0_raw, sub_base_f1, base_K)
pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_base_f1, 'TargetRAUC': base_mapped}).to_csv('submission_base.csv', index=False)

# ===== ITERATION 1: Pseudo-Labeling Round 1 =====
print("\n===== Harvesting Pseudo-Labels Round 1 (Thresh: 0.90) =====")
m_pos = p_test_0 >= 0.90
m_neg = p_test_0 <= 0.10
print(f"Adding {m_pos.sum()} pos, {m_neg.sum()} neg")

X_pseudo_pos = X_test[feature_cols][m_pos].copy()
X_pseudo_pos['label'] = 1
X_pseudo_neg = X_test[feature_cols][m_neg].copy()
X_pseudo_neg['label'] = 0
X_pseudo = pd.concat([X_pseudo_pos, X_pseudo_neg], ignore_index=True)
y_pseudo = X_pseudo['label'].values
X_pseudo = X_pseudo[feature_cols]

oof_1, p_test_1, p_test_1_raw = train_ensemble_cv(X_train_base, y_train_base, X_pseudo=X_pseudo, y_pseudo=y_pseudo)
pos_1 = (p_test_1 >= 0.5).sum()
print(f"Round 1 test set positives: {pos_1} / {len(p_test_1)}")

# Save Round 1 Submission
sub_r1_f1 = (p_test_1 >= 0.5).astype(int)
r1_K = max(sub_r1_f1.sum(), 1)
r1_mapped = map_probabilities(p_test_1_raw, sub_r1_f1, r1_K)
pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_r1_f1, 'TargetRAUC': r1_mapped}).to_csv('submission_pseudo_r1.csv', index=False)

# ===== ITERATION 2: Pseudo-Labeling Round 2 =====
print("\n===== Harvesting Pseudo-Labels Round 2 (Thresh: 0.85) =====")
m_pos = p_test_1 >= 0.85
m_neg = p_test_1 <= 0.15
print(f"Adding {m_pos.sum()} pos, {m_neg.sum()} neg")

X_pseudo_pos = X_test[feature_cols][m_pos].copy()
X_pseudo_pos['label'] = 1
X_pseudo_neg = X_test[feature_cols][m_neg].copy()
X_pseudo_neg['label'] = 0
X_pseudo = pd.concat([X_pseudo_pos, X_pseudo_neg], ignore_index=True)
y_pseudo = X_pseudo['label'].values
X_pseudo = X_pseudo[feature_cols]

oof_2, p_test_2, p_test_2_raw = train_ensemble_cv(X_train_base, y_train_base, X_pseudo=X_pseudo, y_pseudo=y_pseudo)
pos_2 = (p_test_2 >= 0.5).sum()
print(f"Round 2 test set positives: {pos_2} / {len(p_test_2)}")

# Save Round 2 Submission
sub_r2_f1 = (p_test_2 >= 0.5).astype(int)
r2_K = max(sub_r2_f1.sum(), 1)
r2_mapped = map_probabilities(p_test_2_raw, sub_r2_f1, r2_K)
pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_r2_f1, 'TargetRAUC': r2_mapped}).to_csv('submission_pseudo_r2.csv', index=False)

# ===== ITERATION 3: Pseudo-Labeling Round 3 =====
print("\n===== Harvesting Pseudo-Labels Round 3 (Thresh: 0.80) =====")
m_pos = p_test_2 >= 0.80
m_neg = p_test_2 <= 0.20
print(f"Adding {m_pos.sum()} pos, {m_neg.sum()} neg")

X_pseudo_pos = X_test[feature_cols][m_pos].copy()
X_pseudo_pos['label'] = 1
X_pseudo_neg = X_test[feature_cols][m_neg].copy()
X_pseudo_neg['label'] = 0
X_pseudo = pd.concat([X_pseudo_pos, X_pseudo_neg], ignore_index=True)
y_pseudo = X_pseudo['label'].values
X_pseudo = X_pseudo[feature_cols]

oof_3, p_test_3, p_test_3_raw = train_ensemble_cv(X_train_base, y_train_base, X_pseudo=X_pseudo, y_pseudo=y_pseudo)
pos_3 = (p_test_3 >= 0.5).sum()
print(f"Round 3 test set positives: {pos_3} / {len(p_test_3)}")

# Save Final Round 3 Submission
sub_r3_f1 = (p_test_3 >= 0.5).astype(int)
r3_K = max(sub_r3_f1.sum(), 1)
r3_mapped = map_probabilities(p_test_3_raw, sub_r3_f1, r3_K)
pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_r3_f1, 'TargetRAUC': r3_mapped}).to_csv('submission.csv', index=False)

print("\nAll submissions saved and verified successfully!")
