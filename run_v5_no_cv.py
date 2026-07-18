"""
Operation Premiere Place v5 — No CV Pipeline (~5x speedup over run_v5_fast)
=======================================================================
Based on run_v5_fast, but:
1. No cross validation is performed. Models are trained directly on all available data.
2. Estimators/iterations set to 1200, matching run_v5_fast config.
3. In-sample predictions are used for Platt scaling (Logistic Regression) calibration 
   and ensemble weights optimization.
4. Pseudo-labeling is also done without cross validation, training on the combined dataset.
"""
import pandas as pd
import numpy as np
import random
from scipy import stats as scipy_stats
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
N_ESTIMATORS = 1200

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
def get_pattern(row):
    pat = []
    for m in range(1, 13):
        ms = f'{m:02d}'
        has_s1 = (row[f'VH_{ms}'] != -9999) and (row[f'VV_{ms}'] != -9999)
        has_s2 = (row[f'blue_{ms}'] != -9999) and (row[f'green_{ms}'] != -9999) and \
                  (row[f'red_{ms}'] != -9999) and (row[f'nir_{ms}'] != -9999) and \
                  (row[f'swir1_{ms}'] != -9999)
        if has_s1 or has_s2:
            pat.append(m)
    return pat

print("Extracting test missingness patterns...")
test_patterns = [get_pattern(row) for _, row in test.iterrows()]

def mask_train_like_test(seed):
    rng = random.Random(seed)
    tm = train.replace(-9999, np.nan).copy()
    for idx in range(len(tm)):
        pat = rng.choice(test_patterns)
        for m in range(1, 13):
            if m not in pat:
                ms = f'{m:02d}'
                for b in BANDS:
                    tm.at[idx, f'{b}_{ms}'] = np.nan
    return tm

# ===================================================================
# 2. Enhanced NaN-preserving feature extraction
# ===================================================================
def extract_features(df):
    feats = []
    for _, row in df.iterrows():
        f = {}
        series = {k: [] for k in
                  ['mndwi', 'ndvi', 'sar', 'evi', 'sdwi', 'sabi', 'mci',
                   'cdom', 'twobda', 'ndsi', 'awei', 'vh_vv', 'nira_ndvi',
                   'rvi', 'vh_vv_ratio', 'water_product', 'awei_mndwi_diff']}
        observed_months = []

        for m in range(1, 13):
            ms = f'{m:02d}'
            blue = row[f'blue_{ms}']; green = row[f'green_{ms}']
            red = row[f'red_{ms}']; nir = row[f'nir_{ms}']
            nira = row[f'nira_{ms}']; swir1 = row[f'swir1_{ms}']
            swir2 = row[f'swir2_{ms}']; vh = row[f'VH_{ms}']
            vv = row[f'VV_{ms}']; re1 = row[f're1_{ms}']; re2 = row[f're2_{ms}']

            band_vals = [blue, green, red, nir, nira, swir1, swir2, vh, vv, re1, re2]
            if any(pd.isna(v) for v in band_vals):
                for k in series:
                    series[k].append(np.nan)
                continue

            observed_months.append(m)
            vh_lin = 10 ** (vh / 10); vv_lin = 10 ** (vv / 10)
            
            mndwi_val = (green - swir1) / (green + swir1 + 1e-8)
            ndvi_val = (nir - red) / (nir + red + 1e-8)
            sar_val = vh_lin + vv_lin
            
            series['mndwi'].append(mndwi_val)
            series['ndvi'].append(ndvi_val)
            series['sar'].append(sar_val)
            series['evi'].append(2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-8))
            series['sdwi'].append(np.log(10.0 * vv_lin * vh_lin + 1e-8) - 8.0)
            series['sabi'].append((nir - red) / (green + blue + 1e-8))
            series['cdom'].append(green / (red + 1e-8))
            series['mci'].append(re1 - red - 0.5333 * (re2 - red))
            series['twobda'].append(re1 / (red + 1e-8))
            series['ndsi'].append((green - swir2) / (green + swir2 + 1e-8))
            series['awei'].append(4.0 * (green - swir1) - (0.25 * nir + 2.75 * swir2))
            series['vh_vv'].append(vh - vv)
            series['nira_ndvi'].append((nira - red) / (nira + red + 1e-8))
            
            # SAR features
            series['rvi'].append(4 * vh_lin / (vh_lin + vv_lin + 1e-8))
            series['vh_vv_ratio'].append(vh_lin / (vv_lin + 1e-8))
            
            # Cross-index interactions
            series['water_product'].append(mndwi_val * (1 - ndvi_val))
            awei_val = 4.0 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
            series['awei_mndwi_diff'].append(awei_val - mndwi_val)

        arrs = {k: np.array(v, dtype=float) for k, v in series.items()}

        def pstats(a, name):
            valid = a[~np.isnan(a)]
            if len(valid) == 0:
                for suf in ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95',
                            'min', 'max', 'range', 'std', 'frac_pos',
                            'iqr', 'kurtosis', 'skew', 'cv']:
                    f[f'{name}_{suf}'] = 0.0
                return
            s = np.sort(valid)
            f[f'{name}_p5'] = np.percentile(s, 5)
            f[f'{name}_p10'] = np.percentile(s, 10)
            f[f'{name}_p25'] = np.percentile(s, 25)
            f[f'{name}_p50'] = np.percentile(s, 50)
            f[f'{name}_p75'] = np.percentile(s, 75)
            f[f'{name}_p90'] = np.percentile(s, 90)
            f[f'{name}_p95'] = np.percentile(s, 95)
            f[f'{name}_min'] = s[0]
            f[f'{name}_max'] = s[-1]
            f[f'{name}_range'] = s[-1] - s[0]
            f[f'{name}_std'] = np.std(valid)
            f[f'{name}_frac_pos'] = np.mean(valid > 0)
            f[f'{name}_iqr'] = f[f'{name}_p75'] - f[f'{name}_p25']
            f[f'{name}_kurtosis'] = scipy_stats.kurtosis(valid) if len(valid) >= 4 else 0.0
            f[f'{name}_skew'] = scipy_stats.skew(valid) if len(valid) >= 3 else 0.0
            mean_val = np.mean(valid)
            f[f'{name}_cv'] = np.std(valid) / (abs(mean_val) + 1e-8)

        for name, arr in arrs.items():
            pstats(arr, name)

        for name in ['mndwi', 'ndvi', 'sar', 'ndsi', 'rvi', 'water_product']:
            valid_idx = np.where(~np.isnan(arrs[name]))[0]
            if len(valid_idx) >= 2:
                vals = arrs[name][valid_idx]
                diffs = np.diff(vals)
                f[f'{name}_grad_mean'] = np.mean(diffs)
                f[f'{name}_grad_std'] = np.std(diffs)
                f[f'{name}_grad_max'] = np.max(np.abs(diffs))
            else:
                f[f'{name}_grad_mean'] = 0.0
                f[f'{name}_grad_std'] = 0.0
                f[f'{name}_grad_max'] = 0.0

        # Lag-1 autocorrelation
        for name in ['mndwi', 'ndvi', 'sar', 'sdwi']:
            valid_idx = np.where(~np.isnan(arrs[name]))[0]
            if len(valid_idx) >= 4:
                vals = arrs[name][valid_idx]
                if np.std(vals) > 1e-10:
                    autocorr = np.corrcoef(vals[:-1], vals[1:])[0, 1]
                    f[f'{name}_autocorr'] = autocorr if np.isfinite(autocorr) else 0.0
                else:
                    f[f'{name}_autocorr'] = 1.0
                f[f'{name}_masd'] = np.mean(np.abs(np.diff(vals)))
            else:
                f[f'{name}_autocorr'] = 0.0
                f[f'{name}_masd'] = 0.0

        # Water frequency and correlations
        mndwi_valid = arrs['mndwi'][~np.isnan(arrs['mndwi'])]
        sar_valid = arrs['sar'][~np.isnan(arrs['sar'])]
        ndvi_valid = arrs['ndvi'][~np.isnan(arrs['ndvi'])]
        ndsi_valid = arrs['ndsi'][~np.isnan(arrs['ndsi'])]
        sdwi_valid = arrs['sdwi'][~np.isnan(arrs['sdwi'])]

        f['water_freq_strict'] = np.mean((mndwi_valid > 0.5) & (sar_valid < 0.1)) if len(mndwi_valid) else 0.0
        f['water_freq_sdwi'] = np.mean(sdwi_valid > -1.5) if len(sdwi_valid) else 0.0
        f['mndwi_sar_corr'] = np.corrcoef(mndwi_valid, sar_valid)[0, 1] if len(mndwi_valid) > 1 else 0.0
        f['mndwi_ndvi_corr'] = np.corrcoef(mndwi_valid, ndvi_valid)[0, 1] if len(mndwi_valid) > 1 else 0.0
        f['ndsi_mndwi_corr'] = np.corrcoef(ndsi_valid, mndwi_valid)[0, 1] if len(ndsi_valid) > 1 else 0.0
        f['water_score'] = f['mndwi_p25'] * (1 - min(f['sar_p25'], 1))
        f['water_floor'] = f['mndwi_p10'] - f['sar_p10']
        f['water_consist'] = f['mndwi_frac_pos'] * (1 - f['sar_frac_pos'])

        rvi_valid = arrs['rvi'][~np.isnan(arrs['rvi'])]
        if len(rvi_valid) > 1 and len(mndwi_valid) > 1 and len(rvi_valid) == len(mndwi_valid):
            f['rvi_mndwi_corr'] = np.corrcoef(rvi_valid, mndwi_valid)[0, 1]
        else:
            f['rvi_mndwi_corr'] = 0.0
        
        if len(mndwi_valid) > 0 and len(ndvi_valid) > 0 and len(mndwi_valid) == len(ndvi_valid):
            f['water_perm'] = np.mean((mndwi_valid > 0) & (ndvi_valid < 0.3))
        else:
            f['water_perm'] = 0.0

        # Missingness metadata
        n_obs = len(observed_months)
        f['n_months_observed'] = n_obs
        f['first_observed_month'] = observed_months[0] if n_obs else 0
        f['last_observed_month'] = observed_months[-1] if n_obs else 0
        f['block_length'] = (observed_months[-1] - observed_months[0] + 1) if n_obs else 0
        f['obs_month_mid'] = np.mean(observed_months) if n_obs else 0.0
        f['obs_density'] = n_obs / (f['block_length'] + 1e-8) if f['block_length'] > 0 else 0.0

        feats.append(f)
    return pd.DataFrame(feats)

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
# 4. Training loop — No CV, direct training on all data
# ===================================================================
y_train_base = train['label'].values
n_test = len(test)
test_nan = test.replace(-9999, np.nan)

all_test_preds_base = []
all_train_final = []
cached_data = {}

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*60}\n  SEED {seed} ({seed_idx+1}/{len(SEEDS)})\n{'='*60}")

    train_masked = mask_train_like_test(seed)
    X_train = extract_features(train_masked)
    X_test = extract_features(test_nan)
    feature_cols = list(X_train.columns)
    
    # Clean features
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if seed_idx == 0:
        print(f"  Feature count: {len(feature_cols)}")
    cached_data[seed] = (X_train, X_test, feature_cols)

    # LightGBM
    base_lgb = lgb.LGBMClassifier(
        random_state=seed, n_estimators=N_ESTIMATORS, learning_rate=0.01,
        max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
    base_lgb.fit(X_train, y_train_base)
    train_lgb_raw = base_lgb.predict_proba(X_train)[:, 1]
    test_lgb_raw = base_lgb.predict_proba(X_test[feature_cols])[:, 1]

    # CatBoost
    base_cb = CatBoostClassifier(
        random_seed=seed, iterations=N_ESTIMATORS, learning_rate=0.01,
        depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0,
        l2_leaf_reg=5, bagging_temperature=0.5, subsample=0.8)
    base_cb.fit(X_train, y_train_base, verbose=False)
    train_cb_raw = base_cb.predict_proba(X_train)[:, 1]
    test_cb_raw = base_cb.predict_proba(X_test[feature_cols])[:, 1]

    # XGBoost
    base_xgb = xgb.XGBClassifier(
        random_state=seed, n_estimators=N_ESTIMATORS, learning_rate=0.01,
        max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss',
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5)
    base_xgb.fit(X_train, y_train_base, verbose=False)
    train_xgb_raw = base_xgb.predict_proba(X_train)[:, 1]
    test_xgb_raw = base_xgb.predict_proba(X_test[feature_cols])[:, 1]

    # Fit Platt Scaling Calibrators on complete raw train predictions
    cal_lgb = LogisticRegression(C=1e5).fit(train_lgb_raw.reshape(-1, 1), y_train_base)
    cal_cb  = LogisticRegression(C=1e5).fit(train_cb_raw.reshape(-1, 1), y_train_base)
    cal_xgb = LogisticRegression(C=1e5).fit(train_xgb_raw.reshape(-1, 1), y_train_base)
    
    train_lgb = cal_lgb.predict_proba(train_lgb_raw.reshape(-1, 1))[:, 1]
    train_cb  = cal_cb.predict_proba(train_cb_raw.reshape(-1, 1))[:, 1]
    train_xgb = cal_xgb.predict_proba(train_xgb_raw.reshape(-1, 1))[:, 1]
    
    test_lgb = cal_lgb.predict_proba(test_lgb_raw.reshape(-1, 1))[:, 1]
    test_cb  = cal_cb.predict_proba(test_cb_raw.reshape(-1, 1))[:, 1]
    test_xgb = cal_xgb.predict_proba(test_xgb_raw.reshape(-1, 1))[:, 1]

    # Finer blend weights (step 0.05)
    best_score, best_weights = 0, (1/3, 1/3, 1/3)
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            w3 = 1 - w1 - w2
            if w3 < -0.001:
                continue
            w3 = max(w3, 0)
            train_ens = w1 * train_lgb + w2 * train_cb + w3 * train_xgb
            s = 0.6 * f1_score(y_train_base, train_ens >= 0.5) + 0.4 * roc_auc_score(y_train_base, train_ens)
            if s > best_score:
                best_score, best_weights = s, (w1, w2, w3)

    w1, w2, w3 = best_weights
    train_final = w1 * train_lgb + w2 * train_cb + w3 * train_xgb
    all_train_final.append(train_final)
    print(f"  Weights: LGBM={w1:.2f}, CB={w2:.2f}, XGB={w3:.2f} | In-sample Train Score={best_score:.5f}")

    p_test = w1 * test_lgb + w2 * test_cb + w3 * test_xgb
    all_test_preds_base.append(p_test)

p_test_avg = np.mean(all_test_preds_base, axis=0)
train_avg = np.mean(all_train_final, axis=0)
train_score = 0.6 * f1_score(y_train_base, train_avg >= 0.5) + 0.4 * roc_auc_score(y_train_base, train_avg)
print(f"\n{'='*60}\n  TRAIN SCORE (averaged across seeds): {train_score:.5f}\n{'='*60}")

# ===================================================================
# 5. Prior-shift correction — base model
# ===================================================================
p_test_corrected, estimated_test_prior = saerens_prior_correction(p_test_avg, TRAIN_PRIOR)
print(f"\nEstimated test-set prior: {estimated_test_prior:.4f} (train: {TRAIN_PRIOR:.4f})")
print(f"Positives @0.5 before correction: {(p_test_avg >= 0.5).sum()} / {n_test}")
print(f"Positives @0.5 after correction:  {(p_test_corrected >= 0.5).sum()} / {n_test}")

sub_f1 = (p_test_corrected >= 0.5).astype(int)
sub_base = pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_f1, 'TargetRAUC': p_test_corrected})
sub_base.to_csv('submission_base_v5_no_cv.csv', index=False)
print(f"Saved submission_base_v5_no_cv.csv with {sub_f1.sum()} positives.")

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

        X_tr = pd.concat([X_train, X_pseudo], ignore_index=True)
        y_tr = np.concatenate([y_train_base, y_pseudo])

        # LightGBM
        base_lgb = lgb.LGBMClassifier(
            random_state=seed, n_estimators=N_ESTIMATORS, learning_rate=0.01,
            max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=15,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
        base_lgb.fit(X_tr, y_tr)
        train_lgb_raw = base_lgb.predict_proba(X_train)[:, 1]
        test_lgb_raw = base_lgb.predict_proba(X_test[feature_cols])[:, 1]

        # CatBoost
        base_cb = CatBoostClassifier(
            random_seed=seed, iterations=N_ESTIMATORS, learning_rate=0.01,
            depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0,
            l2_leaf_reg=5, bagging_temperature=0.5, subsample=0.8)
        base_cb.fit(X_tr, y_tr, verbose=False)
        train_cb_raw = base_cb.predict_proba(X_train)[:, 1]
        test_cb_raw = base_cb.predict_proba(X_test[feature_cols])[:, 1]

        # XGBoost
        base_xgb = xgb.XGBClassifier(
            random_state=seed, n_estimators=N_ESTIMATORS, learning_rate=0.01,
            max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss',
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=5)
        base_xgb.fit(X_tr, y_tr, verbose=False)
        train_xgb_raw = base_xgb.predict_proba(X_train)[:, 1]
        test_xgb_raw = base_xgb.predict_proba(X_test[feature_cols])[:, 1]

        # Platt Scaling
        cal_lgb = LogisticRegression(C=1e5).fit(train_lgb_raw.reshape(-1, 1), y_train_base)
        cal_cb  = LogisticRegression(C=1e5).fit(train_cb_raw.reshape(-1, 1), y_train_base)
        cal_xgb = LogisticRegression(C=1e5).fit(train_xgb_raw.reshape(-1, 1), y_train_base)
        
        train_lgb = cal_lgb.predict_proba(train_lgb_raw.reshape(-1, 1))[:, 1]
        train_cb  = cal_cb.predict_proba(train_cb_raw.reshape(-1, 1))[:, 1]
        train_xgb = cal_xgb.predict_proba(train_xgb_raw.reshape(-1, 1))[:, 1]
        
        test_lgb = cal_lgb.predict_proba(test_lgb_raw.reshape(-1, 1))[:, 1]
        test_cb  = cal_cb.predict_proba(test_cb_raw.reshape(-1, 1))[:, 1]
        test_xgb = cal_xgb.predict_proba(test_xgb_raw.reshape(-1, 1))[:, 1]

        best_score, best_weights = 0, (1/3, 1/3, 1/3)
        for w1 in np.arange(0, 1.01, 0.05):
            for w2 in np.arange(0, 1.01 - w1, 0.05):
                w3 = 1 - w1 - w2
                if w3 < -0.001: continue
                w3 = max(w3, 0)
                train_ens = w1 * train_lgb + w2 * train_cb + w3 * train_xgb
                s = 0.6 * f1_score(y_train_base, train_ens >= 0.5) + 0.4 * roc_auc_score(y_train_base, train_ens)
                if s > best_score:
                    best_score, best_weights = s, (w1, w2, w3)
        w1, w2, w3 = best_weights
        print(f"  PL Weights: LGBM={w1:.2f}, CB={w2:.2f}, XGB={w3:.2f} | In-sample Train Score={best_score:.5f}")
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
sub_r1.to_csv('submission_pl_r1_v5_no_cv.csv', index=False)
print(f"Saved submission_pl_r1_v5_no_cv.csv with {sub_r1_f1.sum()} positives.")

# Round 2
p_r2 = run_pseudo_labeling(p_r1, cached_data, 0.85, 0.15, "Round 2")
sub_r2_f1 = (p_r2 >= 0.5).astype(int)
sub_r2 = pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_r2_f1, 'TargetRAUC': p_r2})
sub_r2.to_csv('submission_pl_r2_v5_no_cv.csv', index=False)
print(f"Saved submission_pl_r2_v5_no_cv.csv with {sub_r2_f1.sum()} positives.")

# Round 3
p_r3 = run_pseudo_labeling(p_r2, cached_data, 0.80, 0.20, "Round 3")
sub_r3_f1 = (p_r3 >= 0.5).astype(int)
sub_r3 = pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_r3_f1, 'TargetRAUC': p_r3})
sub_r3.to_csv('submission_pl_r3_v5_no_cv.csv', index=False)
print(f"Saved submission_pl_r3_v5_no_cv.csv with {sub_r3_f1.sum()} positives.")

# Summary
print(f"\n{'='*60}")
print(f"  OPERATION PREMIERE PLACE (NO CV EDITION) COMPLETE!")
print(f"{'='*60}")
print(f"Files generated:")
print(f"  submission_base_v5_no_cv.csv      - Base model + Saerens")
print(f"  submission_pl_r1_v5_no_cv.csv     - + Pseudo-labeling R1 (0.90)")
print(f"  submission_pl_r2_v5_no_cv.csv     - + Pseudo-labeling R2 (0.85)")
print(f"  submission_pl_r3_v5_no_cv.csv     - + Pseudo-labeling R3 (0.80)")
