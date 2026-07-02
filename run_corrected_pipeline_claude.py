"""
Operation Ecrasement v3 -- Honest CV + Prior-Shift Correction
================================================================
Fixes vs v2, in order of expected impact on the CV/LB gap:

1. CALIBRATION LEAKAGE (biggest fix): v2 fit CalibratedClassifierCV(cv='prefit')
   on the validation fold and then scored on that *same* fold -> inflated OOF.
   Here calibration uses internal cross-fitting (cv=3, method='sigmoid') on the
   training portion only. The validation fold is never touched by the calibrator
   before being scored. Your honest CV number will drop from v2's number -- that
   is expected and correct, not a regression.

2. CLASS-PRIOR SHIFT: the challenge explicitly states the test set likely has a
   higher pond-positive rate than the 40% in train, and that you should account
   for this (without moving the 0.5 threshold, which is forbidden). We add an
   unsupervised prior-shift correction (Saerens/EM, "Adjusting the outputs of a
   classifier to new a priori probabilities") that recalibrates probabilities
   towards their own estimated test-set prior. This is applied AFTER model
   calibration, BEFORE the fixed 0.5 threshold -- it changes the probabilities,
   not the decision rule.

3. IMPUTATION-INDUCED SIGNAL: v2 imputed raw bands with KNNImputer *before*
   computing ratio indices, meaning missing months got fabricated band values
   that were then turned into fabricated NDVI/MNDWI/etc. Here ratio indices are
   computed directly from the raw (masked) bands with NaNs preserved, and
   aggregated with nan-aware percentiles. No band is ever invented. Explicit
   missingness/position features are added instead (n_months_observed,
   first_observed_month, last_observed_month, block_length).

4. Removed the no-op probability remapping (map_probabilities) -- it was
   mathematically a no-op for AUC (monotonic within/([across the 0.5 split))
   but added complexity. We submit the corrected probability directly.

5. Pseudo-labeling is kept but moved to run only on top of the corrected,
   prior-adjusted probabilities, and is clearly separated so you can compare
   "base corrected" vs "base + pseudo-label" honestly before deciding to submit
   the pseudo-labeled version.

Run this after installing lightgbm, xgboost, catboost (not verified to run
end-to-end in this environment -- those packages were not available here).
Validate on your machine before submitting.
"""
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
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
# 1. Observed-month patterns (unchanged logic from v2, needed for
#    realistic train-time masking that mimics the real test masking)
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
    """Apply a realistic (contiguous-block) masking pattern borrowed from a
    real test row to every train row. No imputation happens here -- masked
    values become NaN and stay NaN through feature extraction."""
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
# 2. Feature extraction directly from raw (NaN-preserving) bands.
#    No cross-sample imputation -- ratio indices and their summary
#    stats are computed only from genuinely observed months.
# ===================================================================
def extract_features(df):
    feats = []
    for _, row in df.iterrows():
        f = {}
        series = {k: [] for k in
                  ['mndwi', 'ndvi', 'sar', 'evi', 'sdwi', 'sabi', 'mci',
                   'cdom', 'twobda', 'ndsi', 'awei', 'vh_vv', 'nira_ndvi']}
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
            series['mndwi'].append((green - swir1) / (green + swir1 + 1e-8))
            series['ndvi'].append((nir - red) / (nir + red + 1e-8))
            series['sar'].append(vh_lin + vv_lin)
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

        arrs = {k: np.array(v, dtype=float) for k, v in series.items()}

        def pstats(a, name):
            valid = a[~np.isnan(a)]
            if len(valid) == 0:
                # should not happen given block masking always leaves >=4 months,
                # but guard anyway
                for suf in ['p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95',
                            'min', 'max', 'range', 'std', 'frac_pos']:
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

        for name, arr in arrs.items():
            pstats(arr, name)

        # temporal gradients computed only across consecutive *observed* months
        for name in ['mndwi', 'ndvi', 'sar', 'ndsi']:
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

        # --- missingness / observation-window metadata (new) ---
        n_obs = len(observed_months)
        f['n_months_observed'] = n_obs
        f['first_observed_month'] = observed_months[0] if n_obs else 0
        f['last_observed_month'] = observed_months[-1] if n_obs else 0
        f['block_length'] = (observed_months[-1] - observed_months[0] + 1) if n_obs else 0
        f['obs_month_mid'] = np.mean(observed_months) if n_obs else 0.0

        feats.append(f)
    return pd.DataFrame(feats)

# ===================================================================
# 3. Saerens/EM prior-shift correction (unsupervised).
#    Iteratively re-estimates the test-set prior from the model's own
#    (calibrated) probabilities and rescales them to match it.
#    Reference: Saerens, Latinne & Decaestecker (2002),
#    "Adjusting the outputs of a classifier to new a priori probabilities".
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
# 4. Training loop -- honest calibration (internal cross-fit, not
#    fit-and-score on the same fold), multi-seed, multi-model.
# ===================================================================
y_train_base = train['label'].values
n_test = len(test)
test_nan = test.replace(-9999, np.nan)

all_test_preds_base = []
all_oof_final = []
cached_data = {}

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*60}\n  SEED {seed} ({seed_idx+1}/{len(SEEDS)})\n{'='*60}")

    train_masked = mask_train_like_test(seed)
    X_train = extract_features(train_masked)
    X_test = extract_features(test_nan)
    feature_cols = list(X_train.columns)
    if seed_idx == 0:
        print(f"  Feature count: {len(feature_cols)}")
    cached_data[seed] = (X_train, X_test, feature_cols)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    oof_lgb = np.zeros(len(y_train_base))
    oof_cb = np.zeros(len(y_train_base))
    oof_xgb = np.zeros(len(y_train_base))
    test_preds_lgb, test_preds_cb, test_preds_xgb = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_base)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train_base[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train_base[val_idx]

        # --- LightGBM, calibrated with internal cross-fitting on X_tr only ---
        base_lgb = lgb.LGBMClassifier(random_state=seed + fold, n_estimators=800,
                                       learning_rate=0.02, max_depth=6,
                                       scale_pos_weight=1.5, verbose=-1, min_child_samples=10)
        cal_lgb = CalibratedClassifierCV(base_lgb, method='sigmoid', cv=3)
        cal_lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = cal_lgb.predict_proba(X_val)[:, 1]
        test_preds_lgb.append(cal_lgb.predict_proba(X_test[feature_cols])[:, 1])

        # --- CatBoost ---
        base_cb = CatBoostClassifier(random_seed=seed + fold, iterations=800,
                                      learning_rate=0.02, depth=6,
                                      auto_class_weights='Balanced', thread_count=-1, verbose=0)
        cal_cb = CalibratedClassifierCV(base_cb, method='sigmoid', cv=3)
        cal_cb.fit(X_tr, y_tr)
        oof_cb[val_idx] = cal_cb.predict_proba(X_val)[:, 1]
        test_preds_cb.append(cal_cb.predict_proba(X_test[feature_cols])[:, 1])

        # --- XGBoost ---
        base_xgb = xgb.XGBClassifier(random_state=seed + fold, n_estimators=800,
                                      learning_rate=0.02, max_depth=6,
                                      scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss')
        cal_xgb = CalibratedClassifierCV(base_xgb, method='sigmoid', cv=3)
        cal_xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = cal_xgb.predict_proba(X_val)[:, 1]
        test_preds_xgb.append(cal_xgb.predict_proba(X_test[feature_cols])[:, 1])

    # Blend weights chosen on the now-honest OOF
    best_score, best_weights = 0, (1/3, 1/3, 1/3)
    for w1 in np.linspace(0, 1, 11):
        for w2 in np.linspace(0, 1 - w1, 11):
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
            oof_ens = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
            s = 0.6 * f1_score(y_train_base, oof_ens >= 0.5) + 0.4 * roc_auc_score(y_train_base, oof_ens)
            if s > best_score:
                best_score, best_weights = s, (w1, w2, w3)

    w1, w2, w3 = best_weights
    oof_final = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
    all_oof_final.append(oof_final)
    print(f"  Weights: LGBM={w1:.2f}, CB={w2:.2f}, XGB={w3:.2f} | Honest CV={best_score:.5f}")

    p_test = (w1 * np.mean(test_preds_lgb, axis=0) +
              w2 * np.mean(test_preds_cb, axis=0) +
              w3 * np.mean(test_preds_xgb, axis=0))
    all_test_preds_base.append(p_test)

p_test_avg = np.mean(all_test_preds_base, axis=0)
oof_avg = np.mean(all_oof_final, axis=0)
honest_cv = 0.6 * f1_score(y_train_base, oof_avg >= 0.5) + 0.4 * roc_auc_score(y_train_base, oof_avg)
print(f"\n{'='*60}\n  HONEST CV (post-fix, averaged across seeds): {honest_cv:.5f}\n{'='*60}")
print("Compare this number to your real Zindi LB score, not the old 0.97 -- "
      "it should now be a realistic estimate.")

# ===================================================================
# 5. Prior-shift correction on the test predictions only
#    (OOF stays anchored to the true train prior since that's correct
#    for the train distribution; we only adjust where we suspect shift).
# ===================================================================
p_test_corrected, estimated_test_prior = saerens_prior_correction(p_test_avg, TRAIN_PRIOR)
print(f"\nEstimated test-set positive prior via Saerens/EM: {estimated_test_prior:.4f} "
      f"(train prior was {TRAIN_PRIOR:.4f})")
print(f"Positives @0.5 before correction: {(p_test_avg >= 0.5).sum()} / {n_test}")
print(f"Positives @0.5 after correction:  {(p_test_corrected >= 0.5).sum()} / {n_test}")

sub_f1 = (p_test_corrected >= 0.5).astype(int)
sub_base = pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_f1, 'TargetRAUC': p_test_corrected})
sub_base.to_csv('submission_base_corrected.csv', index=False)
print(f"Saved submission_base_corrected.csv with {sub_f1.sum()} positives.")

# ===================================================================
# 6. Optional pseudo-labeling round, built on the corrected probabilities.
#    Run this only after confirming submission_base_corrected.csv beats
#    your previous LB score -- don't stack an unvalidated technique on
#    top of another.
# ===================================================================
RUN_PSEUDO_LABELING = True
if RUN_PSEUDO_LABELING:
    print(f"\n{'='*60}\n  PSEUDO-LABELING (on prior-corrected probabilities)\n{'='*60}")
    m_pos = p_test_corrected >= 0.90
    m_neg = p_test_corrected <= 0.10
    print(f"Pseudo-positive: {m_pos.sum()}, Pseudo-negative: {m_neg.sum()}")

    all_test_preds_pl = []
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  PL Seed {seed} ({seed_idx+1}/{len(SEEDS)})...")
        X_train, X_test, feature_cols = cached_data[seed]
        X_pseudo = pd.concat([X_test[feature_cols][m_pos], X_test[feature_cols][m_neg]], ignore_index=True)
        y_pseudo = np.concatenate([np.ones(m_pos.sum()), np.zeros(m_neg.sum())])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
        test_preds_lgb, test_preds_cb, test_preds_xgb = [], [], []
        oof_lgb = np.zeros(len(y_train_base)); oof_cb = np.zeros(len(y_train_base)); oof_xgb = np.zeros(len(y_train_base))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_base)):
            X_tr = pd.concat([X_train.iloc[tr_idx], X_pseudo], ignore_index=True)
            y_tr = np.concatenate([y_train_base[tr_idx], y_pseudo])
            X_val, y_val = X_train.iloc[val_idx], y_train_base[val_idx]

            base_lgb = lgb.LGBMClassifier(random_state=seed + fold, n_estimators=800, learning_rate=0.02,
                                           max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=10)
            cal_lgb = CalibratedClassifierCV(base_lgb, method='sigmoid', cv=3)
            cal_lgb.fit(X_tr, y_tr)
            oof_lgb[val_idx] = cal_lgb.predict_proba(X_val)[:, 1]
            test_preds_lgb.append(cal_lgb.predict_proba(X_test[feature_cols])[:, 1])

            base_cb = CatBoostClassifier(random_seed=seed + fold, iterations=800, learning_rate=0.02,
                                          depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0)
            cal_cb = CalibratedClassifierCV(base_cb, method='sigmoid', cv=3)
            cal_cb.fit(X_tr, y_tr)
            oof_cb[val_idx] = cal_cb.predict_proba(X_val)[:, 1]
            test_preds_cb.append(cal_cb.predict_proba(X_test[feature_cols])[:, 1])

            base_xgb = xgb.XGBClassifier(random_state=seed + fold, n_estimators=800, learning_rate=0.02,
                                          max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss')
            cal_xgb = CalibratedClassifierCV(base_xgb, method='sigmoid', cv=3)
            cal_xgb.fit(X_tr, y_tr)
            oof_xgb[val_idx] = cal_xgb.predict_proba(X_val)[:, 1]
            test_preds_xgb.append(cal_xgb.predict_proba(X_test[feature_cols])[:, 1])

        best_score, best_weights = 0, (1/3, 1/3, 1/3)
        for w1 in np.linspace(0, 1, 11):
            for w2 in np.linspace(0, 1 - w1, 11):
                w3 = 1 - w1 - w2
                if w3 < 0:
                    continue
                oof_ens = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
                s = 0.6 * f1_score(y_train_base, oof_ens >= 0.5) + 0.4 * roc_auc_score(y_train_base, oof_ens)
                if s > best_score:
                    best_score, best_weights = s, (w1, w2, w3)
        w1, w2, w3 = best_weights
        print(f"  PL Weights: LGBM={w1:.2f}, CB={w2:.2f}, XGB={w3:.2f} | Honest CV={best_score:.5f}")
        p_test = (w1 * np.mean(test_preds_lgb, axis=0) +
                  w2 * np.mean(test_preds_cb, axis=0) +
                  w3 * np.mean(test_preds_xgb, axis=0))
        all_test_preds_pl.append(p_test)

    p_test_pl = np.mean(all_test_preds_pl, axis=0)
    # re-apply prior correction to the pseudo-labeled predictions too
    p_test_pl_corrected, prior_pl = saerens_prior_correction(p_test_pl, TRAIN_PRIOR)
    print(f"\nEstimated test prior after PL round: {prior_pl:.4f}")

    sub_pl_f1 = (p_test_pl_corrected >= 0.5).astype(int)
    sub_final = pd.DataFrame({'ID': test['ID'], 'TargetF1': sub_pl_f1, 'TargetRAUC': p_test_pl_corrected})
    sub_final.to_csv('submission_pseudo_corrected.csv', index=False)
    print(f"Saved submission_pseudo_corrected.csv with {sub_pl_f1.sum()} positives.")

print("\n" + "=" * 60)
print("  DONE. Compare submission_base_corrected.csv against your")
print("  previous best on the LB before trusting the pseudo-label round.")
print("=" * 60)
