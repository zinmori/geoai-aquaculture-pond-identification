"""
Operation Premiere Place v5 — Fast 1st Place Pipeline (~25x speedup)
=======================================================================
Based on v4, but optimized for speed:
1. Platt scaling (Logistic Regression) on out-of-fold (OOF) predictions
   instead of nested CalibratedClassifierCV(cv=3) retraining.
2. Direct early stopping for CatBoost and XGBoost on CPU.
3. Feature engineering and pseudo-labeling logic remains 100% identical
   to preserve the 0.9389 winning performance.
"""
import pandas as pd
import numpy as np
import random
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

RNG_SEED = 42
SEEDS = [42, 123, 456, 789, 1337]
BANDS = ['blue', 'green', 'red', 'nir', 'nira', 'swir1', 'swir2', 're1', 're2', 're3', 'VH', 'VV']

print("Loading data...")
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

feature_cols_raw = [c for c in train.columns if c not in ['ID', 'label']]
TRAIN_PRIOR = train['label'].mean()
print(f"Train prior (fraction positive): {TRAIN_PRIOR:.4f}")

# ===================================================================
# 1. Observed-month patterns for realistic masking
# ===================================================================
print("Extracting test missingness patterns...")
test_observed = np.zeros((len(test), 12), dtype=bool)
for m in range(1, 13):
    ms = f'{m:02d}'
    has_s1 = (test[f'VH_{ms}'].values != -9999) & (test[f'VV_{ms}'].values != -9999)
    has_s2 = (test[f'blue_{ms}'].values != -9999) & (test[f'green_{ms}'].values != -9999) & \
             (test[f'red_{ms}'].values != -9999) & (test[f'nir_{ms}'].values != -9999) & \
             (test[f'swir1_{ms}'].values != -9999)
    test_observed[:, m - 1] = has_s1 | has_s2
test_patterns = [list(np.where(row)[0] + 1) for row in test_observed]

train_nan = train.replace(-9999, np.nan)

def mask_train_like_test(seed):
    rng = random.Random(seed)
    tm = train_nan.copy()
    chosen_patterns = [rng.choice(test_patterns) for _ in range(len(tm))]
    
    mask_nan = np.ones((len(tm), 12), dtype=bool)
    for idx, pat in enumerate(chosen_patterns):
        for m in pat:
            mask_nan[idx, m - 1] = False
            
    for m in range(1, 13):
        ms = f'{m:02d}'
        rows_to_mask = np.where(mask_nan[:, m - 1])[0]
        if len(rows_to_mask) > 0:
            cols = [f'{b}_{ms}' for b in BANDS]
            tm.loc[tm.index[rows_to_mask], cols] = np.nan
    return tm

# ===================================================================
# 2. Enhanced NaN-preserving feature extraction
# ===================================================================
def get_row_corr(x, y, valid_mask):
    n_v = np.sum(valid_mask, axis=1)
    x_clean = np.where(valid_mask, x, 0.0)
    y_clean = np.where(valid_mask, y, 0.0)
    mean_x = np.sum(x_clean, axis=1) / (n_v + 1e-12)
    mean_y = np.sum(y_clean, axis=1) / (n_v + 1e-12)
    xm = np.where(valid_mask, x - mean_x[:, None], 0.0)
    ym = np.where(valid_mask, y - mean_y[:, None], 0.0)
    cov = np.sum(xm * ym, axis=1)
    var_x = np.sum(xm**2, axis=1)
    var_y = np.sum(ym**2, axis=1)
    den = np.sqrt(var_x * var_y)
    corr = np.zeros(len(x))
    valid_corr = (n_v > 1) & (den > 1e-10)
    corr[valid_corr] = cov[valid_corr] / den[valid_corr]
    return corr

def extract_features(df):
    N = len(df)
    
    blue = np.empty((N, 12))
    green = np.empty((N, 12))
    red = np.empty((N, 12))
    nir = np.empty((N, 12))
    nira = np.empty((N, 12))
    swir1 = np.empty((N, 12))
    swir2 = np.empty((N, 12))
    vh = np.empty((N, 12))
    vv = np.empty((N, 12))
    re1 = np.empty((N, 12))
    re2 = np.empty((N, 12))

    for m in range(1, 13):
        ms = f'{m:02d}'
        blue[:, m - 1] = df[f'blue_{ms}'].values
        green[:, m - 1] = df[f'green_{ms}'].values
        red[:, m - 1] = df[f'red_{ms}'].values
        nir[:, m - 1] = df[f'nir_{ms}'].values
        nira[:, m - 1] = df[f'nira_{ms}'].values
        swir1[:, m - 1] = df[f'swir1_{ms}'].values
        swir2[:, m - 1] = df[f'swir2_{ms}'].values
        vh[:, m - 1] = df[f'VH_{ms}'].values
        vv[:, m - 1] = df[f'VV_{ms}'].values
        re1[:, m - 1] = df[f're1_{ms}'].values
        re2[:, m - 1] = df[f're2_{ms}'].values

    valid_mask = ~(
        np.isnan(blue) | np.isnan(green) | np.isnan(red) | np.isnan(nir) |
        np.isnan(nira) | np.isnan(swir1) | np.isnan(swir2) | np.isnan(vh) |
        np.isnan(vv) | np.isnan(re1) | np.isnan(re2)
    )

    vh_lin = 10.0 ** (vh / 10.0)
    vv_lin = 10.0 ** (vv / 10.0)
    
    mndwi = (green - swir1) / (green + swir1 + 1e-8)
    ndvi = (nir - red) / (nir + red + 1e-8)
    sar = vh_lin + vv_lin
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-8)
    sdwi = np.log(10.0 * vv_lin * vh_lin + 1e-8) - 8.0
    sabi = (nir - red) / (green + blue + 1e-8)
    cdom = green / (red + 1e-8)
    mci = re1 - red - 0.5333 * (re2 - red)
    twobda = re1 / (red + 1e-8)
    ndsi = (green - swir2) / (green + swir2 + 1e-8)
    awei = 4.0 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    vh_vv = vh - vv
    nira_ndvi = (nira - red) / (nira + red + 1e-8)
    rvi = 4.0 * vh_lin / (vh_lin + vv_lin + 1e-8)
    vh_vv_ratio = vh_lin / (vv_lin + 1e-8)
    water_product = mndwi * (1.0 - ndvi)
    awei_mndwi_diff = awei - mndwi

    series = {
        'mndwi': mndwi, 'ndvi': ndvi, 'sar': sar, 'evi': evi, 'sdwi': sdwi,
        'sabi': sabi, 'cdom': cdom, 'mci': mci, 'twobda': twobda, 'ndsi': ndsi,
        'awei': awei, 'vh_vv': vh_vv, 'nira_ndvi': nira_ndvi, 'rvi': rvi,
        'vh_vv_ratio': vh_vv_ratio, 'water_product': water_product, 'awei_mndwi_diff': awei_mndwi_diff
    }
    
    for arr in series.values():
        arr[~valid_mask] = np.nan

    feats_dict = {}
    n_v = np.sum(valid_mask, axis=1)

    for name, arr in series.items():
        with np.errstate(all='ignore'):
            p5 = np.nanpercentile(arr, 5, axis=1)
            p10 = np.nanpercentile(arr, 10, axis=1)
            p25 = np.nanpercentile(arr, 25, axis=1)
            p50 = np.nanpercentile(arr, 50, axis=1)
            p75 = np.nanpercentile(arr, 75, axis=1)
            p90 = np.nanpercentile(arr, 90, axis=1)
            p95 = np.nanpercentile(arr, 95, axis=1)
            min_val = np.nanmin(arr, axis=1)
            max_val = np.nanmax(arr, axis=1)
            std_val = np.nanstd(arr, axis=1)
            frac_pos = np.nanmean(np.where(np.isnan(arr), np.nan, arr > 0), axis=1)
            
            mean_val = np.nanmean(arr, axis=1, keepdims=True)
            diff = arr - mean_val
            diff[np.isnan(diff)] = 0.0
            n_v_col = n_v[:, None]
            m2 = np.sum(diff**2, axis=1, keepdims=True) / n_v_col
            m3 = np.sum(diff**3, axis=1, keepdims=True) / n_v_col
            m4 = np.sum(diff**4, axis=1, keepdims=True) / n_v_col
            
            std_val_check = np.sqrt(m2.squeeze(-1))
            skew = np.where((n_v >= 3) & (std_val_check > 1e-10), m3.squeeze(-1) / (m2.squeeze(-1)**1.5), 0.0)
            kurtosis = np.where((n_v >= 4) & (std_val_check > 1e-10), m4.squeeze(-1) / (m2.squeeze(-1)**2) - 3.0, 0.0)
            
            cv = std_val / (np.abs(mean_val.squeeze(-1)) + 1e-8)

        p5 = np.nan_to_num(p5, nan=0.0)
        p10 = np.nan_to_num(p10, nan=0.0)
        p25 = np.nan_to_num(p25, nan=0.0)
        p50 = np.nan_to_num(p50, nan=0.0)
        p75 = np.nan_to_num(p75, nan=0.0)
        p90 = np.nan_to_num(p90, nan=0.0)
        p95 = np.nan_to_num(p95, nan=0.0)
        min_val = np.nan_to_num(min_val, nan=0.0)
        max_val = np.nan_to_num(max_val, nan=0.0)
        std_val = np.nan_to_num(std_val, nan=0.0)
        frac_pos = np.nan_to_num(frac_pos, nan=0.0)
        skew = np.nan_to_num(skew, nan=0.0)
        kurtosis = np.nan_to_num(kurtosis, nan=0.0)
        cv = np.nan_to_num(cv, nan=0.0)
        
        range_val = max_val - min_val
        iqr = p75 - p25

        feats_dict[f'{name}_p5'] = p5
        feats_dict[f'{name}_p10'] = p10
        feats_dict[f'{name}_p25'] = p25
        feats_dict[f'{name}_p50'] = p50
        feats_dict[f'{name}_p75'] = p75
        feats_dict[f'{name}_p90'] = p90
        feats_dict[f'{name}_p95'] = p95
        feats_dict[f'{name}_min'] = min_val
        feats_dict[f'{name}_max'] = max_val
        feats_dict[f'{name}_range'] = range_val
        feats_dict[f'{name}_std'] = std_val
        feats_dict[f'{name}_frac_pos'] = frac_pos
        feats_dict[f'{name}_iqr'] = iqr
        feats_dict[f'{name}_kurtosis'] = kurtosis
        feats_dict[f'{name}_skew'] = skew
        feats_dict[f'{name}_cv'] = cv

    grad_names = ['mndwi', 'ndvi', 'sar', 'ndsi', 'rvi', 'water_product']
    grad_mean = {name: np.zeros(N) for name in grad_names}
    grad_std = {name: np.zeros(N) for name in grad_names}
    grad_max = {name: np.zeros(N) for name in grad_names}
    
    autocorr_names = ['mndwi', 'ndvi', 'sar', 'sdwi']
    autocorr = {name: np.zeros(N) for name in autocorr_names}
    masd = {name: np.zeros(N) for name in autocorr_names}

    for i in range(N):
        valid_idx = np.where(valid_mask[i])[0]
        n_obs_i = len(valid_idx)
        if n_obs_i >= 2:
            for name in grad_names:
                vals = series[name][i, valid_idx]
                diffs = vals[1:] - vals[:-1]
                grad_mean[name][i] = np.mean(diffs)
                grad_std[name][i] = np.std(diffs)
                grad_max[name][i] = np.max(np.abs(diffs))
        if n_obs_i >= 4:
            for name in autocorr_names:
                vals = series[name][i, valid_idx]
                v_std = np.std(vals)
                if v_std > 1e-10:
                    x = vals[:-1]
                    y = vals[1:]
                    mx = np.mean(x)
                    my = np.mean(y)
                    xm, ym = x - mx, y - my
                    r_num = np.sum(xm * ym)
                    r_den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
                    if r_den > 1e-10:
                        r = r_num / r_den
                        autocorr[name][i] = r if np.isfinite(r) else 0.0
                    else:
                        autocorr[name][i] = 0.0
                else:
                    autocorr[name][i] = 1.0
                masd[name][i] = np.mean(np.abs(vals[1:] - vals[:-1]))

    for name in grad_names:
        feats_dict[f'{name}_grad_mean'] = grad_mean[name]
        feats_dict[f'{name}_grad_std'] = grad_std[name]
        feats_dict[f'{name}_grad_max'] = grad_max[name]
        
    for name in autocorr_names:
        feats_dict[f'{name}_autocorr'] = autocorr[name]
        feats_dict[f'{name}_masd'] = masd[name]

    mndwi_valid = series['mndwi']
    sar_valid = series['sar']
    ndvi_valid = series['ndvi']
    ndsi_valid = series['ndsi']
    sdwi_valid = series['sdwi']
    rvi_valid = series['rvi']

    with np.errstate(all='ignore'):
        feats_dict['water_freq_strict'] = np.nan_to_num(np.nanmean(np.where(np.isnan(mndwi_valid), np.nan, (mndwi_valid > 0.5) & (sar_valid < 0.1)), axis=1), nan=0.0)
        feats_dict['water_freq_sdwi'] = np.nan_to_num(np.nanmean(np.where(np.isnan(sdwi_valid), np.nan, sdwi_valid > -1.5), axis=1), nan=0.0)
        
        feats_dict['mndwi_sar_corr'] = get_row_corr(mndwi_valid, sar_valid, valid_mask)
        feats_dict['mndwi_ndvi_corr'] = get_row_corr(mndwi_valid, ndvi_valid, valid_mask)
        feats_dict['ndsi_mndwi_corr'] = get_row_corr(ndsi_valid, mndwi_valid, valid_mask)
        
        feats_dict['water_score'] = feats_dict['mndwi_p25'] * (1 - np.minimum(feats_dict['sar_p25'], 1.0))
        feats_dict['water_floor'] = feats_dict['mndwi_p10'] - feats_dict['sar_p10']
        feats_dict['water_consist'] = feats_dict['mndwi_frac_pos'] * (1 - feats_dict['sar_frac_pos'])
        
        feats_dict['rvi_mndwi_corr'] = get_row_corr(rvi_valid, mndwi_valid, valid_mask)
        feats_dict['water_perm'] = np.nan_to_num(np.nanmean(np.where(np.isnan(mndwi_valid), np.nan, (mndwi_valid > 0) & (ndvi_valid < 0.3)), axis=1), nan=0.0)

    first_obs = np.where(n_v > 0, np.argmax(valid_mask, axis=1) + 1, 0)
    last_obs = np.where(n_v > 0, 12 - np.argmax(valid_mask[:, ::-1], axis=1), 0)
    block_len = np.where(n_v > 0, last_obs - first_obs + 1, 0)
    
    obs_mid = np.zeros(N)
    obs_mid_val = np.sum(valid_mask * np.arange(1, 13), axis=1) / n_v
    obs_mid[n_v > 0] = obs_mid_val[n_v > 0]
    
    obs_dens = np.zeros(N)
    obs_dens_val = n_v / (block_len + 1e-8)
    obs_dens[block_len > 0] = obs_dens_val[block_len > 0]

    feats_dict['n_months_observed'] = n_v
    feats_dict['first_observed_month'] = first_obs
    feats_dict['last_observed_month'] = last_obs
    feats_dict['block_length'] = block_len
    feats_dict['obs_month_mid'] = obs_mid
    feats_dict['obs_density'] = obs_dens

    return pd.DataFrame(feats_dict)

# ===================================================================
# 3. Saerens/EM prior-shift correction
# ===================================================================
def saerens_prior_correction(p_test, train_prior, max_iter=100, tol=1e-6):
    p_test = np.clip(p_test, 1e-6, 1 - 1e-6)
    prior_new = train_prior
    for _ in range(max_iter):
        w = prior_new * (1 - train_prior) / (train_prior * (1 - prior_new) + 1e-12)
        num = p_test * w
        p_adj = num / (num + (1 - p_test) + 1e-12)
        prior_updated = p_adj.mean()
        if abs(prior_updated - prior_new) < tol:
            prior_new = prior_updated
            break
        prior_new = prior_updated
    w = prior_new * (1 - train_prior) / (train_prior * (1 - prior_new) + 1e-12)
    num = p_test * w
    p_final = num / (num + (1 - p_test) + 1e-12)
    return p_final, prior_new

# ===================================================================
# 4. Training loop — OOF Platt scaling calibration
# ===================================================================
y_train_base = train['label'].values
n_test = len(test)
test_nan = test.replace(-9999, np.nan)

# Extract test features once outside of all loops
print("Extracting features for static test dataset...")
X_test_static = extract_features(test_nan)
X_test_static = X_test_static.replace([np.inf, -np.inf], np.nan).fillna(0)

all_test_preds_base = []
all_oof_final = []
cached_data = {}

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*60}\n  SEED {seed} ({seed_idx+1}/{len(SEEDS)})\n{'='*60}")

    train_masked = mask_train_like_test(seed)
    X_train = extract_features(train_masked)
    X_test = X_test_static.copy()
    feature_cols = list(X_train.columns)
    
    # Clean features
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if seed_idx == 0:
        print(f"  Feature count: {len(feature_cols)}")
    cached_data[seed] = (X_train, X_test, feature_cols)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store raw OOF and raw Test predictions
    oof_lgb_raw = np.zeros(len(y_train_base))
    oof_cb_raw  = np.zeros(len(y_train_base))
    oof_xgb_raw = np.zeros(len(y_train_base))
    
    test_lgb_raw_folds = []
    test_cb_raw_folds  = []
    test_xgb_raw_folds = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_base)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train_base[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train_base[val_idx]

        # LightGBM
        base_lgb = lgb.LGBMClassifier(
            random_state=seed + fold, n_estimators=1200, learning_rate=0.01,
            max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=15,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
        base_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb_raw[val_idx] = base_lgb.predict_proba(X_val)[:, 1]
        test_lgb_raw_folds.append(base_lgb.predict_proba(X_test[feature_cols])[:, 1])

        # CatBoost
        base_cb = CatBoostClassifier(
            random_seed=seed + fold, iterations=1200, learning_rate=0.01,
            depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0,
            l2_leaf_reg=5, bagging_temperature=0.5, subsample=0.8)
        base_cb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        oof_cb_raw[val_idx] = base_cb.predict_proba(X_val)[:, 1]
        test_cb_raw_folds.append(base_cb.predict_proba(X_test[feature_cols])[:, 1])

        # XGBoost
        base_xgb = xgb.XGBClassifier(
            random_state=seed + fold, n_estimators=1200, learning_rate=0.01,
            max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss',
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=5, early_stopping_rounds=50)
        base_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb_raw[val_idx] = base_xgb.predict_proba(X_val)[:, 1]
        test_xgb_raw_folds.append(base_xgb.predict_proba(X_test[feature_cols])[:, 1])

    # Fit Platt Scaling Calibrators on complete raw OOF arrays
    cal_lgb = LogisticRegression(C=1e5).fit(oof_lgb_raw.reshape(-1, 1), y_train_base)
    cal_cb  = LogisticRegression(C=1e5).fit(oof_cb_raw.reshape(-1, 1), y_train_base)
    cal_xgb = LogisticRegression(C=1e5).fit(oof_xgb_raw.reshape(-1, 1), y_train_base)
    
    oof_lgb = cal_lgb.predict_proba(oof_lgb_raw.reshape(-1, 1))[:, 1]
    oof_cb  = cal_cb.predict_proba(oof_cb_raw.reshape(-1, 1))[:, 1]
    oof_xgb = cal_xgb.predict_proba(oof_xgb_raw.reshape(-1, 1))[:, 1]
    
    test_lgb = cal_lgb.predict_proba(np.mean(test_lgb_raw_folds, axis=0).reshape(-1, 1))[:, 1]
    test_cb  = cal_cb.predict_proba(np.mean(test_cb_raw_folds, axis=0).reshape(-1, 1))[:, 1]
    test_xgb = cal_xgb.predict_proba(np.mean(test_xgb_raw_folds, axis=0).reshape(-1, 1))[:, 1]

    # Finer blend weights (step 0.05)
    best_score, best_weights = 0, (1/3, 1/3, 1/3)
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            w3 = 1 - w1 - w2
            if w3 < -0.001:
                continue
            w3 = max(w3, 0)
            oof_ens = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
            s = 0.6 * f1_score(y_train_base, oof_ens >= 0.5) + 0.4 * roc_auc_score(y_train_base, oof_ens)
            if s > best_score:
                best_score, best_weights = s, (w1, w2, w3)

    w1, w2, w3 = best_weights
    oof_final = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
    all_oof_final.append(oof_final)
    print(f"  Weights: LGBM={w1:.2f}, CB={w2:.2f}, XGB={w3:.2f} | Honest CV={best_score:.5f}")

    p_test = w1 * test_lgb + w2 * test_cb + w3 * test_xgb
    all_test_preds_base.append(p_test)

p_test_avg = np.mean(all_test_preds_base, axis=0)
oof_avg = np.mean(all_oof_final, axis=0)
honest_cv = 0.6 * f1_score(y_train_base, oof_avg >= 0.5) + 0.4 * roc_auc_score(y_train_base, oof_avg)
print(f"\n{'='*60}\n  HONEST CV (averaged across seeds): {honest_cv:.5f}\n{'='*60}")

# ===================================================================
# 5. Prior-shift correction — base model
# ===================================================================
p_test_corrected, estimated_test_prior = saerens_prior_correction(p_test_avg, TRAIN_PRIOR)
print(f"\nEstimated test-set prior: {estimated_test_prior:.4f} (train: {TRAIN_PRIOR:.4f})")
print(f"Positives @0.5 before correction: {(p_test_avg >= 0.5).sum()} / {n_test}")
print(f"Positives @0.5 after correction:  {(p_test_corrected >= 0.5).sum()} / {n_test}")

sub_f1 = (p_test_corrected >= 0.5).astype(int)
sub_base = pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_f1, 'TargetRAUC': p_test_corrected})
sub_base.to_csv('submission_base_v5_fast.csv', index=False)
print(f"Saved submission_base_v5_fast.csv with {sub_f1.sum()} positives.")

# ===================================================================
# 6. PSEUDO-LABELING Rounds
# ===================================================================
def run_pseudo_labeling(p_ref, cached_data, threshold_pos, threshold_neg, round_name):
    print(f"\n{'='*60}\n  PSEUDO-LABELING {round_name} (pos>={threshold_pos}, neg<={threshold_neg})\n{'='*60}")
    m_pos = p_ref >= threshold_pos
    m_neg = p_ref <= threshold_neg
    print(f"Pseudo-positive: {m_pos.sum()}, Pseudo-negative: {m_neg.sum()}")

    all_test_preds_pl = []
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  PL Seed {seed} ({seed_idx+1}/{len(SEEDS)})...")
        X_train, X_test, feature_cols = cached_data[seed]
        X_pseudo = pd.concat([X_test[feature_cols][m_pos], X_test[feature_cols][m_neg]], ignore_index=True)
        y_pseudo = np.concatenate([np.ones(m_pos.sum()), np.zeros(m_neg.sum())])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        oof_lgb_raw = np.zeros(len(y_train_base))
        oof_cb_raw  = np.zeros(len(y_train_base))
        oof_xgb_raw = np.zeros(len(y_train_base))
        
        test_lgb_raw_folds = []
        test_cb_raw_folds  = []
        test_xgb_raw_folds = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_base)):
            X_tr = pd.concat([X_train.iloc[tr_idx], X_pseudo], ignore_index=True)
            y_tr = np.concatenate([y_train_base[tr_idx], y_pseudo])
            X_val, y_val = X_train.iloc[val_idx], y_train_base[val_idx]

            # LightGBM
            base_lgb = lgb.LGBMClassifier(
                random_state=seed + fold, n_estimators=1200, learning_rate=0.01,
                max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=15,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
            base_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_lgb_raw[val_idx] = base_lgb.predict_proba(X_val)[:, 1]
            test_lgb_raw_folds.append(base_lgb.predict_proba(X_test[feature_cols])[:, 1])

            # CatBoost
            base_cb = CatBoostClassifier(
                random_seed=seed + fold, iterations=1200, learning_rate=0.01,
                depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0,
                l2_leaf_reg=5, bagging_temperature=0.5, subsample=0.8)
            base_cb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            oof_cb_raw[val_idx] = base_cb.predict_proba(X_val)[:, 1]
            test_cb_raw_folds.append(base_cb.predict_proba(X_test[feature_cols])[:, 1])

            # XGBoost
            base_xgb = xgb.XGBClassifier(
                random_state=seed + fold, n_estimators=1200, learning_rate=0.01,
                max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss',
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                min_child_weight=5, early_stopping_rounds=50)
            base_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof_xgb_raw[val_idx] = base_xgb.predict_proba(X_val)[:, 1]
            test_xgb_raw_folds.append(base_xgb.predict_proba(X_test[feature_cols])[:, 1])

        # Platt Scaling
        cal_lgb = LogisticRegression(C=1e5).fit(oof_lgb_raw.reshape(-1, 1), y_train_base)
        cal_cb  = LogisticRegression(C=1e5).fit(oof_cb_raw.reshape(-1, 1), y_train_base)
        cal_xgb = LogisticRegression(C=1e5).fit(oof_xgb_raw.reshape(-1, 1), y_train_base)
        
        oof_lgb = cal_lgb.predict_proba(oof_lgb_raw.reshape(-1, 1))[:, 1]
        oof_cb  = cal_cb.predict_proba(oof_cb_raw.reshape(-1, 1))[:, 1]
        oof_xgb = cal_xgb.predict_proba(oof_xgb_raw.reshape(-1, 1))[:, 1]
        
        test_lgb = cal_lgb.predict_proba(np.mean(test_lgb_raw_folds, axis=0).reshape(-1, 1))[:, 1]
        test_cb  = cal_cb.predict_proba(np.mean(test_cb_raw_folds, axis=0).reshape(-1, 1))[:, 1]
        test_xgb = cal_xgb.predict_proba(np.mean(test_xgb_raw_folds, axis=0).reshape(-1, 1))[:, 1]

        best_score, best_weights = 0, (1/3, 1/3, 1/3)
        for w1 in np.arange(0, 1.01, 0.05):
            for w2 in np.arange(0, 1.01 - w1, 0.05):
                w3 = 1 - w1 - w2
                if w3 < -0.001: continue
                w3 = max(w3, 0)
                oof_ens = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
                s = 0.6 * f1_score(y_train_base, oof_ens >= 0.5) + 0.4 * roc_auc_score(y_train_base, oof_ens)
                if s > best_score:
                    best_score, best_weights = s, (w1, w2, w3)
        w1, w2, w3 = best_weights
        print(f"  PL Weights: LGBM={w1:.2f}, CB={w2:.2f}, XGB={w3:.2f} | Honest CV={best_score:.5f}")
        p_test = w1 * test_lgb + w2 * test_cb + w3 * test_xgb
        all_test_preds_pl.append(p_test)

    p_test_pl = np.mean(all_test_preds_pl, axis=0)
    p_test_pl_corrected, prior_pl = saerens_prior_correction(p_test_pl, TRAIN_PRIOR)
    print(f"\nEstimated test prior after {round_name}: {prior_pl:.4f}")
    return p_test_pl_corrected

# Round 1
p_r1 = run_pseudo_labeling(p_test_corrected, cached_data, 0.90, 0.10, "Round 1")
sub_r1_f1 = (p_r1 >= 0.5).astype(int)
sub_r1 = pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_r1_f1, 'TargetRAUC': p_r1})
sub_r1.to_csv('submission_pl_r1_v5_fast.csv', index=False)
print(f"Saved submission_pl_r1_v5_fast.csv with {sub_r1_f1.sum()} positives.")

# Round 2
p_r2 = run_pseudo_labeling(p_r1, cached_data, 0.85, 0.15, "Round 2")
sub_r2_f1 = (p_r2 >= 0.5).astype(int)
sub_r2 = pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_r2_f1, 'TargetRAUC': p_r2})
sub_r2.to_csv('submission_pl_r2_v5_fast.csv', index=False)
print(f"Saved submission_pl_r2_v5_fast.csv with {sub_r2_f1.sum()} positives.")

# Round 3
p_r3 = run_pseudo_labeling(p_r2, cached_data, 0.80, 0.20, "Round 3")
sub_r3_f1 = (p_r3 >= 0.5).astype(int)
sub_r3 = pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_r3_f1, 'TargetRAUC': p_r3})
sub_r3.to_csv('submission_pl_r3_v5_fast.csv', index=False)
print(f"Saved submission_pl_r3_v5_fast.csv with {sub_r3_f1.sum()} positives.")

# Summary
print(f"\n{'='*60}")
print(f"  OPERATION PREMIERE PLACE (FAST EDITION) COMPLETE!")
print(f"{'='*60}")
print(f"Files generated:")
print(f"  submission_base_v5_fast.csv      - Base model + Saerens")
print(f"  submission_pl_r1_v5_fast.csv     - + Pseudo-labeling R1 (0.90)")
print(f"  submission_pl_r2_v5_fast.csv     - + Pseudo-labeling R2 (0.85)")
