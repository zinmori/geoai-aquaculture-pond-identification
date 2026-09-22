import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

RNG_SEED = 42
SEEDS = [42, 123, 456, 789, 1337, 2024, 7, 99]
BANDS = ['blue', 'green', 'red', 'nir', 'nira', 'swir1', 'swir2', 're1', 're2', 're3', 'VH', 'VV']
OPT = ['blue', 'green', 'red', 'nir', 'nira', 'swir1', 'swir2', 're1', 're2', 're3']
SAR = ['VH', 'VV']

print("Loading data...")
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

TRAIN_PRIOR = train['label'].mean()
print(f"Train prior (fraction positive): {TRAIN_PRIOR:.4f}")

# ===================================================================
# 1. Motifs d'observation reels, PAR CAPTEUR
# ===================================================================
print("Extracting test missingness patterns (per sensor)...")
test_opt = np.zeros((len(test), 12), dtype=bool)
test_sar = np.zeros((len(test), 12), dtype=bool)
for m in range(1, 13):
    ms = f'{m:02d}'
    test_sar[:, m - 1] = (test[f'VH_{ms}'].values != -9999) & (test[f'VV_{ms}'].values != -9999)
    test_opt[:, m - 1] = (test[f'blue_{ms}'].values != -9999) & (test[f'green_{ms}'].values != -9999) & \
                         (test[f'red_{ms}'].values != -9999) & (test[f'nir_{ms}'].values != -9999) & \
                         (test[f'swir1_{ms}'].values != -9999)
n_both = int((test_opt & test_sar).sum())
n_s1only = int((test_sar & ~test_opt).sum())
print(f"  {n_both} cellules bi-capteur, {n_s1only} SAR-only "
      f"(v5 jetait ces {n_s1only} cellules radar)")

train_nan = train.replace(-9999, np.nan)


def mask_train_like_test(seed):
    """Une ligne de test tiree au hasard donne son motif d'observation, capteur
    par capteur : l'optique et le radar peuvent manquer independamment."""
    rng = random.Random(seed)
    tm = train_nan.copy()
    idx = [rng.randrange(len(test)) for _ in range(len(tm))]
    o = test_opt[idx]
    s = test_sar[idx]
    for m in range(1, 13):
        ms = f'{m:02d}'
        rows = np.where(~o[:, m - 1])[0]
        if len(rows):
            tm.loc[tm.index[rows], [f'{b}_{ms}' for b in OPT]] = np.nan
        rows = np.where(~s[:, m - 1])[0]
        if len(rows):
            tm.loc[tm.index[rows], [f'{b}_{ms}' for b in SAR]] = np.nan
    return tm


# ===================================================================
# 2. Extraction de features
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
    a = {b: np.stack([df[f'{b}_{m:02d}'].values for m in range(1, 13)], axis=1).astype(float)
         for b in BANDS}

    # masque optique : les bandes que v5 exigeait, moins le radar
    opt_mask = np.ones((N, 12), dtype=bool)
    for b in ['blue', 'green', 'red', 'nir', 'nira', 'swir1', 'swir2', 're1', 're2']:
        opt_mask &= ~np.isnan(a[b])
    sar_mask = ~np.isnan(a['VH']) & ~np.isnan(a['VV'])

    vh_lin = 10.0 ** (a['VH'] / 10.0)
    vv_lin = 10.0 ** (a['VV'] / 10.0)

    with np.errstate(all='ignore'):
        mndwi = (a['green'] - a['swir1']) / (a['green'] + a['swir1'] + 1e-8)
        ndvi = (a['nir'] - a['red']) / (a['nir'] + a['red'] + 1e-8)
        sar = vh_lin + vv_lin
        evi = 2.5 * (a['nir'] - a['red']) / (a['nir'] + 6.0 * a['red'] - 7.5 * a['blue'] + 1.0 + 1e-8)
        sdwi = np.log(10.0 * vv_lin * vh_lin + 1e-8) - 8.0
        sabi = (a['nir'] - a['red']) / (a['green'] + a['blue'] + 1e-8)
        cdom = a['green'] / (a['red'] + 1e-8)
        mci = a['re1'] - a['red'] - 0.5333 * (a['re2'] - a['red'])
        twobda = a['re1'] / (a['red'] + 1e-8)
        ndsi = (a['green'] - a['swir2']) / (a['green'] + a['swir2'] + 1e-8)
        awei = 4.0 * (a['green'] - a['swir1']) - (0.25 * a['nir'] + 2.75 * a['swir2'])
        vh_vv = a['VH'] - a['VV']
        nira_ndvi = (a['nira'] - a['red']) / (a['nira'] + a['red'] + 1e-8)
        rvi = 4.0 * vh_lin / (vh_lin + vv_lin + 1e-8)
        vh_vv_ratio = vh_lin / (vv_lin + 1e-8)
        water_product = mndwi * (1.0 - ndvi)
        awei_mndwi_diff = awei - mndwi

    series = {
        'mndwi': mndwi, 'ndvi': ndvi, 'sar': sar, 'evi': evi, 'sdwi': sdwi,
        'sabi': sabi, 'cdom': cdom, 'mci': mci, 'twobda': twobda, 'ndsi': ndsi,
        'awei': awei, 'vh_vv': vh_vv, 'nira_ndvi': nira_ndvi, 'rvi': rvi,
        'vh_vv_ratio': vh_vv_ratio, 'water_product': water_product,
        'awei_mndwi_diff': awei_mndwi_diff
    }
    # chaque serie porte le masque de son capteur
    SAR_SERIES = {'sar', 'sdwi', 'vh_vv', 'rvi', 'vh_vv_ratio'}
    smask = {k: (sar_mask if k in SAR_SERIES else opt_mask) for k in series}
    for k in series:
        series[k] = np.where(smask[k], series[k], np.nan)

    feats_dict = {}

    for name, arr in series.items():
        m = smask[name]
        n_s = np.sum(m, axis=1)
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
            n_col = np.maximum(n_s, 1)[:, None]
            m2 = np.sum(diff**2, axis=1, keepdims=True) / n_col
            m3 = np.sum(diff**3, axis=1, keepdims=True) / n_col
            m4 = np.sum(diff**4, axis=1, keepdims=True) / n_col

            std_check = np.sqrt(np.maximum(m2.squeeze(-1), 0.0))
            skew = np.where((n_s >= 3) & (std_check > 1e-10),
                            m3.squeeze(-1) / (m2.squeeze(-1)**1.5), 0.0)
            kurtosis = np.where((n_s >= 4) & (std_check > 1e-10),
                                m4.squeeze(-1) / (m2.squeeze(-1)**2) - 3.0, 0.0)
            cv = std_val / (np.abs(mean_val.squeeze(-1)) + 1e-8)

        for nm, v in (('p5', p5), ('p10', p10), ('p25', p25), ('p50', p50),
                      ('p75', p75), ('p90', p90), ('p95', p95), ('min', min_val),
                      ('max', max_val), ('std', std_val), ('frac_pos', frac_pos),
                      ('skew', skew), ('kurtosis', kurtosis), ('cv', cv)):
            feats_dict[f'{name}_{nm}'] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        feats_dict[f'{name}_range'] = feats_dict[f'{name}_max'] - feats_dict[f'{name}_min']
        feats_dict[f'{name}_iqr'] = feats_dict[f'{name}_p75'] - feats_dict[f'{name}_p25']

    # --- gradients et autocorrelation, sur les mois valides de CHAQUE serie
    grad_names = ['mndwi', 'ndvi', 'sar', 'ndsi', 'rvi', 'water_product']
    autocorr_names = ['mndwi', 'ndvi', 'sar', 'sdwi']
    grad_mean = {n: np.zeros(N) for n in grad_names}
    grad_std = {n: np.zeros(N) for n in grad_names}
    grad_max = {n: np.zeros(N) for n in grad_names}
    autocorr = {n: np.zeros(N) for n in autocorr_names}
    masd = {n: np.zeros(N) for n in autocorr_names}

    for name in set(grad_names) | set(autocorr_names):
        m = smask[name]
        arr = series[name]
        for i in range(N):
            valid_idx = np.where(m[i])[0]
            if len(valid_idx) >= 2 and name in grad_names:
                vals = arr[i, valid_idx]
                diffs = vals[1:] - vals[:-1]
                grad_mean[name][i] = np.mean(diffs)
                grad_std[name][i] = np.std(diffs)
                grad_max[name][i] = np.max(np.abs(diffs))
            if len(valid_idx) >= 4 and name in autocorr_names:
                vals = arr[i, valid_idx]
                if np.std(vals) > 1e-10:
                    x, yv = vals[:-1], vals[1:]
                    xm, ym = x - np.mean(x), yv - np.mean(yv)
                    r_den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
                    if r_den > 1e-10:
                        r = np.sum(xm * ym) / r_den
                        autocorr[name][i] = r if np.isfinite(r) else 0.0
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

    both = opt_mask & sar_mask
    with np.errstate(all='ignore'):
        n_o = np.maximum(np.sum(opt_mask, axis=1), 1)
        n_s = np.maximum(np.sum(sar_mask, axis=1), 1)
        n_b = np.maximum(np.sum(both, axis=1), 1)
        feats_dict['water_freq_strict'] = np.sum(both & (series['mndwi'] > 0.5) & (series['sar'] < 0.1), axis=1) / n_b
        feats_dict['water_freq_sdwi'] = np.sum(sar_mask & (series['sdwi'] > -1.5), axis=1) / n_s

        feats_dict['mndwi_sar_corr'] = get_row_corr(np.nan_to_num(series['mndwi']), np.nan_to_num(series['sar']), both)
        feats_dict['mndwi_ndvi_corr'] = get_row_corr(np.nan_to_num(series['mndwi']), np.nan_to_num(series['ndvi']), opt_mask)
        feats_dict['ndsi_mndwi_corr'] = get_row_corr(np.nan_to_num(series['ndsi']), np.nan_to_num(series['mndwi']), opt_mask)

        feats_dict['water_score'] = feats_dict['mndwi_p25'] * (1 - np.minimum(feats_dict['sar_p25'], 1.0))
        feats_dict['water_floor'] = feats_dict['mndwi_p10'] - feats_dict['sar_p10']
        feats_dict['water_consist'] = feats_dict['mndwi_frac_pos'] * (1 - feats_dict['sar_frac_pos'])

        feats_dict['rvi_mndwi_corr'] = get_row_corr(np.nan_to_num(series['rvi']), np.nan_to_num(series['mndwi']), both)
        feats_dict['water_perm'] = np.sum(opt_mask & (series['mndwi'] > 0) & (series['ndvi'] < 0.3), axis=1) / n_o

    # --- structure d'observation (v5) sur "un capteur au moins"
    vany = opt_mask | sar_mask
    n_v = np.sum(vany, axis=1)
    first_obs = np.where(n_v > 0, np.argmax(vany, axis=1) + 1, 0)
    last_obs = np.where(n_v > 0, 12 - np.argmax(vany[:, ::-1], axis=1), 0)
    block_len = np.where(n_v > 0, last_obs - first_obs + 1, 0)
    obs_mid = np.zeros(N)
    obs_mid_val = np.sum(vany * np.arange(1, 13), axis=1) / np.maximum(n_v, 1)
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
    # necessaires au decouplage : sans elles le modele ne peut pas distinguer
    # un mois complet d'un mois radar-seul
    feats_dict['n_optical_months'] = np.sum(opt_mask, axis=1).astype(float)
    feats_dict['n_saronly_months'] = np.sum(sar_mask & ~opt_mask, axis=1).astype(float)

    return pd.DataFrame(feats_dict)


# ===================================================================
# 3. Saerens / EM 
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
    return num / (num + (1 - p_test) + 1e-12), prior_new


# ===================================================================
# 4. Entrainement (identique a v5)
# ===================================================================
y_train_base = train['label'].values
n_test = len(test)
test_nan = test.replace(-9999, np.nan)

print("Extracting features for static test dataset...")
X_test_static = extract_features(test_nan).replace([np.inf, -np.inf], np.nan).fillna(0)

all_test_preds_base = []
all_oof_final = []
cached_data = {}


def fit_fold(X_tr, y_tr, X_val, y_val, X_test, feature_cols, seed, fold):
    base_lgb = lgb.LGBMClassifier(
        random_state=seed + fold, n_estimators=1200, learning_rate=0.01,
        max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
    base_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(50, verbose=False)])

    base_cb = CatBoostClassifier(
        random_seed=seed + fold, iterations=1200, learning_rate=0.01,
        depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0,
        l2_leaf_reg=5, bagging_temperature=0.5, subsample=0.8)
    base_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)

    base_xgb = xgb.XGBClassifier(
        random_state=seed + fold, n_estimators=1200, learning_rate=0.01,
        max_depth=6, scale_pos_weight=1.5, n_jobs=-1, eval_metric='logloss',
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=5, early_stopping_rounds=50)
    base_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    return ((base_lgb.predict_proba(X_val)[:, 1], base_cb.predict_proba(X_val)[:, 1],
             base_xgb.predict_proba(X_val)[:, 1]),
            (base_lgb.predict_proba(X_test[feature_cols])[:, 1],
             base_cb.predict_proba(X_test[feature_cols])[:, 1],
             base_xgb.predict_proba(X_test[feature_cols])[:, 1]))


def blend_and_calibrate(oof_raw, test_raw):
    oof_c, test_c = [], []
    for o, t in zip(oof_raw, test_raw):
        cal = LogisticRegression(C=1e5).fit(o.reshape(-1, 1), y_train_base)
        oof_c.append(cal.predict_proba(o.reshape(-1, 1))[:, 1])
        test_c.append(cal.predict_proba(np.mean(t, axis=0).reshape(-1, 1))[:, 1])
    best_score, best_weights = 0, (1 / 3, 1 / 3, 1 / 3)
    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1, 0.05):
            w3 = max(1 - w1 - w2, 0)
            oof_ens = w1 * oof_c[0] + w2 * oof_c[1] + w3 * oof_c[2]
            s = 0.6 * f1_score(y_train_base, oof_ens >= 0.5) + 0.4 * roc_auc_score(y_train_base, oof_ens)
            if s > best_score:
                best_score, best_weights = s, (w1, w2, w3)
    w1, w2, w3 = best_weights
    return (w1 * oof_c[0] + w2 * oof_c[1] + w3 * oof_c[2],
            w1 * test_c[0] + w2 * test_c[1] + w3 * test_c[2],
            best_weights, best_score)


for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*60}\n  SEED {seed} ({seed_idx+1}/{len(SEEDS)})\n{'='*60}")
    X_train = extract_features(mask_train_like_test(seed))
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    feature_cols = list(X_train.columns)
    X_test = X_test_static.copy()
    if seed_idx == 0:
        print(f"  Feature count: {len(feature_cols)}")
    cached_data[seed] = (X_train, X_test, feature_cols)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_raw = [np.zeros(len(y_train_base)) for _ in range(3)]
    test_raw = [[] for _ in range(3)]

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_base)):
        ov, tv = fit_fold(X_train.iloc[tr_idx], y_train_base[tr_idx],
                          X_train.iloc[val_idx], y_train_base[val_idx],
                          X_test, feature_cols, seed, fold)
        for j in range(3):
            oof_raw[j][val_idx] = ov[j]
            test_raw[j].append(tv[j])

    oof_final, p_test, wts, sc = blend_and_calibrate(oof_raw, test_raw)
    all_oof_final.append(oof_final)
    all_test_preds_base.append(p_test)
    print(f"  Weights: LGBM={wts[0]:.2f}, CB={wts[1]:.2f}, XGB={wts[2]:.2f} | CV={sc:.5f}")

p_test_avg = np.mean(all_test_preds_base, axis=0)
oof_avg = np.mean(all_oof_final, axis=0)
honest_cv = 0.6 * f1_score(y_train_base, oof_avg >= 0.5) + 0.4 * roc_auc_score(y_train_base, oof_avg)
print(f"\n{'='*60}\n  CV (moyennee sur {len(SEEDS)} seeds): {honest_cv:.5f}\n{'='*60}")

p_test_corrected, est_prior = saerens_prior_correction(p_test_avg, TRAIN_PRIOR)
print(f"\nPrior test estime: {est_prior:.4f} (train: {TRAIN_PRIOR:.4f})")
print(f"Positifs @0.5 avant correction: {(p_test_avg >= 0.5).sum()} / {n_test}")
print(f"Positifs @0.5 apres correction: {(p_test_corrected >= 0.5).sum()} / {n_test}")

pd.DataFrame({'ID': test['ID'], 'TargetF1': (p_test_corrected >= 0.5).astype(int),
              'TargetRAUC': p_test_corrected}).to_csv('submission_v9_base.csv', index=False)
print(f"-> submission_v9_base.csv ({int((p_test_corrected >= 0.5).sum())} positifs)")


# ===================================================================
# 5. Pseudo-labeling
# ===================================================================
def run_pseudo_labeling(p_ref, threshold_pos, threshold_neg, round_name):
    print(f"\n{'='*60}\n  PSEUDO-LABELING {round_name} (pos>={threshold_pos}, neg<={threshold_neg})\n{'='*60}")
    m_pos = p_ref >= threshold_pos
    m_neg = p_ref <= threshold_neg
    print(f"Pseudo-positifs: {m_pos.sum()}, pseudo-negatifs: {m_neg.sum()}")

    all_test_preds_pl = []
    for seed_idx, seed in enumerate(SEEDS):
        X_train, X_test, feature_cols = cached_data[seed]
        X_pseudo = pd.concat([X_test[feature_cols][m_pos], X_test[feature_cols][m_neg]], ignore_index=True)
        y_pseudo = np.concatenate([np.ones(m_pos.sum()), np.zeros(m_neg.sum())])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_raw = [np.zeros(len(y_train_base)) for _ in range(3)]
        test_raw = [[] for _ in range(3)]

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_base)):
            X_tr = pd.concat([X_train.iloc[tr_idx], X_pseudo], ignore_index=True)
            y_tr = np.concatenate([y_train_base[tr_idx], y_pseudo])
            ov, tv = fit_fold(X_tr, y_tr, X_train.iloc[val_idx], y_train_base[val_idx],
                              X_test, feature_cols, seed, fold)
            for j in range(3):
                oof_raw[j][val_idx] = ov[j]
                test_raw[j].append(tv[j])

        _, p_test, wts, sc = blend_and_calibrate(oof_raw, test_raw)
        all_test_preds_pl.append(p_test)
        print(f"  seed {seed} ({seed_idx+1}/{len(SEEDS)}): LGBM={wts[0]:.2f} CB={wts[1]:.2f} "
              f"XGB={wts[2]:.2f} | CV={sc:.5f}")

    p_pl = np.mean(all_test_preds_pl, axis=0)
    p_pl_corr, prior_pl = saerens_prior_correction(p_pl, TRAIN_PRIOR)
    print(f"Prior estime apres {round_name}: {prior_pl:.4f}")
    return p_pl_corr


p_r1 = run_pseudo_labeling(p_test_corrected, 0.90, 0.10, "Round 1")
pd.DataFrame({'ID': test['ID'], 'TargetF1': (p_r1 >= 0.5).astype(int),
              'TargetRAUC': p_r1}).to_csv('submission_v9_pl_r1.csv', index=False)
print(f"-> submission_v9_pl_r1.csv ({int((p_r1 >= 0.5).sum())} positifs)")

p_r2 = run_pseudo_labeling(p_r1, 0.85, 0.15, "Round 2")
pd.DataFrame({'ID': test['ID'], 'TargetF1': (p_r2 >= 0.5).astype(int),
              'TargetRAUC': p_r2}).to_csv('submission_v9_pl_r2.csv', index=False)
print(f"-> submission_v9_pl_r2.csv ({int((p_r2 >= 0.5).sum())} positifs)")

p_r3 = run_pseudo_labeling(p_r2, 0.80, 0.20, "Round 3")
pd.DataFrame({'ID': test['ID'], 'TargetF1': (p_r3 >= 0.5).astype(int),
              'TargetRAUC': p_r3}).to_csv('submission_v9_pl_r3.csv', index=False)
print(f"-> submission_v9_pl_r3.csv ({int((p_r3 >= 0.5).sum())} positifs)")

np.savez('cache/v9_preds.npz', base=p_test_corrected, r1=p_r1, r2=p_r2, r3=p_r3)
print(f"\n{'='*60}\n  v9 TERMINE\n{'='*60}")
