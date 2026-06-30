"""
Approach A — LB: 0.866
Train on full 12-month data, permanence + percentile features only.
Inference on the real partial test observations.
Key insight: features computed on N months must be the same as on 12 months in meaning.
"""
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")

def extract_permanence_features(row, obs):
    """
    Features that work equally well on 2 months or 12 months.
    Focus: permanence of water signal, not temporal statistics.
    """
    feats = {}
    feats['num_observed'] = len(obs)
    if len(obs) > 0:
        feats['window_size'] = max(obs) - min(obs) + 1
        feats['start_month'] = min(obs)
        feats['end_month'] = max(obs)
    else:
        feats['window_size'] = 0
        feats['start_month'] = 0
        feats['end_month'] = 0

    mndwi_vals, ndwi_vals, ndvi_vals, lswi_vals = [], [], [], []
    awei_vals, sar_sum_vals, sar_diff_vals, rvi_vals = [], [], [], []
    ndre_vals, ndci_vals = [], []

    for m in obs:
        m_str = f'{m:02d}'
        blue  = row[f'blue_{m_str}']
        green = row[f'green_{m_str}']
        red   = row[f'red_{m_str}']
        nir   = row[f'nir_{m_str}']
        swir1 = row[f'swir1_{m_str}']
        swir2 = row[f'swir2_{m_str}']
        vh    = row[f'VH_{m_str}']
        vv    = row[f'VV_{m_str}']
        re1   = row[f're1_{m_str}']
        re2   = row[f're2_{m_str}']

        if blue == -9999 or green == -9999 or red == -9999 or nir == -9999:
            continue
        if swir1 == -9999 or vh == -9999 or vv == -9999:
            continue

        vh_lin = 10.0 ** (vh / 10.0)
        vv_lin = 10.0 ** (vv / 10.0)
        sar_sum = vh_lin + vv_lin

        mndwi = (green - swir1) / (green + swir1 + 1e-8)
        ndwi  = (green - nir)   / (green + nir   + 1e-8)
        ndvi  = (nir   - red)   / (nir   + red   + 1e-8)
        lswi  = (nir   - swir1) / (nir   + swir1 + 1e-8)
        awei  = 4*(green - swir1) - (0.25*nir + 2.75*swir2)
        sar_diff = vv - vh
        rvi_v = 4.0 * vh_lin / (sar_sum + 1e-8)

        ndre = (re2 - re1) / (re2 + re1 + 1e-8) if re1 != -9999 and re2 != -9999 else np.nan
        ndci = (re1 - red) / (re1 + red + 1e-8) if re1 != -9999 else np.nan

        mndwi_vals.append(mndwi); ndwi_vals.append(ndwi); ndvi_vals.append(ndvi)
        lswi_vals.append(lswi); awei_vals.append(awei); sar_sum_vals.append(sar_sum)
        sar_diff_vals.append(sar_diff); rvi_vals.append(rvi_v)
        if not np.isnan(ndre): ndre_vals.append(ndre)
        if not np.isnan(ndci): ndci_vals.append(ndci)

    def perm_stats(vals, name, threshold=0.0):
        if len(vals) == 0:
            for s in ['p10','p25','p50','p75','p90','min','max','range','std','frac_pos']:
                feats[f'{name}_{s}'] = np.nan
            return
        arr = np.array(sorted(vals)); n = len(arr)
        feats[f'{name}_p10']     = arr[max(0, int(n*0.1))]
        feats[f'{name}_p25']     = arr[max(0, int(n*0.25))]
        feats[f'{name}_p50']     = arr[n//2]
        feats[f'{name}_p75']     = arr[min(n-1, int(n*0.75))]
        feats[f'{name}_p90']     = arr[min(n-1, int(n*0.9))]
        feats[f'{name}_min']     = arr[0]
        feats[f'{name}_max']     = arr[-1]
        feats[f'{name}_range']   = arr[-1] - arr[0]
        feats[f'{name}_std']     = np.std(arr)
        feats[f'{name}_frac_pos'] = np.mean(arr > threshold)

    perm_stats(mndwi_vals, 'mndwi', threshold=0.0)
    perm_stats(ndwi_vals,  'ndwi',  threshold=0.0)
    perm_stats(ndvi_vals,  'ndvi',  threshold=0.2)
    perm_stats(lswi_vals,  'lswi',  threshold=0.0)
    perm_stats(awei_vals,  'awei',  threshold=0.0)
    perm_stats(sar_sum_vals,'sar',  threshold=0.04)
    perm_stats(sar_diff_vals,'sar_diff', threshold=0.0)
    perm_stats(rvi_vals,   'rvi',   threshold=0.3)
    perm_stats(ndre_vals,  'ndre',  threshold=0.0)
    perm_stats(ndci_vals,  'ndci',  threshold=0.0)

    if mndwi_vals and sar_sum_vals:
        feats['water_signal_min'] = min(mndwi_vals) - max(sar_sum_vals)
        feats['water_signal_p25'] = feats['mndwi_p25'] - feats['sar_p75']
        feats['combined_water_p50'] = feats['mndwi_p50'] * (1 - min(sar_sum_vals))
    else:
        feats['water_signal_min'] = np.nan
        feats['water_signal_p25'] = np.nan
        feats['combined_water_p50'] = np.nan

    return feats


print("Extracting test features (real partial observations)...")
test_rows = []
for idx, row in test.iterrows():
    obs = [m for m in range(1,13) if row[f'blue_{m:02d}'] != -9999]
    test_rows.append(extract_permanence_features(row, obs))
X_test = pd.DataFrame(test_rows)

print("Extracting train features (full 12 months)...")
train_rows = []
for idx, row in train.iterrows():
    feats = extract_permanence_features(row, list(range(1, 13)))
    feats['label'] = row['label']
    train_rows.append(feats)
X_train_full = pd.DataFrame(train_rows)

y = X_train_full['label']
feature_cols = [c for c in X_train_full.columns if c != 'label']
X_train = X_train_full[feature_cols]
print(f"Train features: {X_train.shape}, Test features: {X_test.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(train)); oof_cb = np.zeros(len(train)); oof_xgb = np.zeros(len(train))

print("\nTraining 5-fold CV...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"  Fold {fold}...")
    X_tr, y_tr = X_train.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X_train.iloc[val_idx], y.iloc[val_idx]

    clf_lgb = lgb.LGBMClassifier(random_state=42, n_estimators=600, learning_rate=0.03,
                                  max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=10)
    clf_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]

    clf_cb = CatBoostClassifier(random_seed=42, iterations=600, learning_rate=0.03,
                                depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0)
    clf_cb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
    oof_cb[val_idx] = clf_cb.predict_proba(X_val)[:, 1]

    clf_xgb = xgb.XGBClassifier(random_state=42, n_estimators=600, learning_rate=0.03,
                                 max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss')
    clf_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = clf_xgb.predict_proba(X_val)[:, 1]

best_score, best_weights, best_cal = 0, None, None
for w1 in np.linspace(0, 1, 11):
    for w2 in np.linspace(0, 1-w1, 11):
        w3 = 1 - w1 - w2
        if w3 < 0: continue
        oof_ens = w1*oof_lgb + w2*oof_cb + w3*oof_xgb
        iso = IsotonicRegression(out_of_bounds='clip'); iso.fit(oof_ens, y)
        cal = iso.predict(oof_ens)
        s = 0.6*f1_score(y, cal>=0.5) + 0.4*roc_auc_score(y, cal)
        if s > best_score: best_score, best_weights, best_cal = s, (w1, w2, w3), cal

w1, w2, w3 = best_weights
print(f"\nBest ensemble weights: LGB={w1:.2f}, CB={w2:.2f}, XGB={w3:.2f}")

print("\nTraining final models on all train data...")
final_lgb = lgb.LGBMClassifier(random_state=42, n_estimators=600, learning_rate=0.03,
                                max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=10)
final_lgb.fit(X_train, y)
p_lgb = final_lgb.predict_proba(X_test[feature_cols])[:, 1]

final_cb = CatBoostClassifier(random_seed=42, iterations=600, learning_rate=0.03,
                              depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0)
final_cb.fit(X_train, y)
p_cb = final_cb.predict_proba(X_test[feature_cols])[:, 1]

final_xgb = xgb.XGBClassifier(random_state=42, n_estimators=600, learning_rate=0.03,
                               max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss')
final_xgb.fit(X_train, y)
p_xgb = final_xgb.predict_proba(X_test[feature_cols])[:, 1]

oof_ens_final = w1*oof_lgb + w2*oof_cb + w3*oof_xgb
iso_final = IsotonicRegression(out_of_bounds='clip'); iso_final.fit(oof_ens_final, y)

p_ens = w1*p_lgb + w2*p_cb + w3*p_xgb
p_cal = iso_final.predict(p_ens)
p_bin = (p_cal >= 0.5).astype(int)

sub = pd.DataFrame({'ID': test['ID'], 'TargetF1': p_bin, 'TargetRAUC': p_cal})
sub.to_csv('submission.csv', index=False)
print("\nSubmission saved to submission.csv")
print("Class distribution:", sub['TargetF1'].value_counts().to_dict())
