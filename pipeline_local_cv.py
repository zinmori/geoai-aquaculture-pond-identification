"""
Proper Local CV Framework that correlates with the Leaderboard.

Key insight: Our current CV (0.98+) is biased because both train and val 
use full 12-month observations. 

This script creates a realistic CV:
1. Split train 80/20 stratified
2. Train model on 80% with full 12 months (as usual)
3. For the 20% validation set, SIMULATE partial windows:
   - Draw a random contiguous window of 3-6 months
   - Start from months 1-9 (matching real test distribution)
   - Extract features from ONLY those months
4. Evaluate on masked validation → this should correlate much better with LB

This is the "gold standard" local evaluation for this problem.
"""
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

train = pd.read_csv('Train.csv')
test  = pd.read_csv('Test.csv')

# ---- Real test window distribution ----
test_windows = []
for idx, row in test.iterrows():
    obs = [m for m in range(1,13) if row[f'blue_{m:02d}'] != -9999]
    if obs:
        test_windows.append((min(obs), len(obs)))

# Build empirical distribution of (start_month, n_obs) from test
tw_arr = np.array(test_windows)
print(f"Test window: mean n_obs={tw_arr[:,1].mean():.2f}, mean start={tw_arr[:,0].mean():.2f}")

def sample_test_window():
    """Sample a window matching the real test distribution."""
    idx = random.randint(0, len(test_windows)-1)
    start, n = test_windows[idx]
    obs = list(range(start, min(start+n, 13)))
    return obs

def compute_monthly_all(row, month):
    ms = f'{month:02d}'
    blue=row[f'blue_{ms}']; green=row[f'green_{ms}']; red=row[f'red_{ms}']
    nir=row[f'nir_{ms}']; swir1=row[f'swir1_{ms}']; swir2=row[f'swir2_{ms}']
    vh=row[f'VH_{ms}']; vv=row[f'VV_{ms}']
    re1=row[f're1_{ms}']; re2=row[f're2_{ms}']
    if any(v==-9999 for v in [blue,green,red,nir,swir1,swir2,vh,vv]): return None
    vh_lin=10**(vh/10); vv_lin=10**(vv/10); sar=vh_lin+vv_lin
    ndre=(re2-re1)/(re2+re1+1e-8) if re1!=-9999 and re2!=-9999 else np.nan
    ndci=(re1-red)/(re1+red+1e-8) if re1!=-9999 else np.nan
    return dict(
        mndwi=(green-swir1)/(green+swir1+1e-8),
        ndwi=(green-nir)/(green+nir+1e-8),
        ndvi=(nir-red)/(nir+red+1e-8),
        lswi=(nir-swir1)/(nir+swir1+1e-8),
        awei=4*(green-swir1)-(0.25*nir+2.75*swir2),
        sar=sar, rvi=4*vh_lin/(sar+1e-8),
        sar_diff=vv-vh, ndre=ndre, ndci=ndci,
        bsi=((swir1+red)-(nir+blue))/((swir1+red)+(nir+blue)+1e-8)
    )

def extract_features(row, obs):
    feats = {}
    feats['num_observed'] = len(obs)
    if obs:
        feats['window_size'] = max(obs)-min(obs)+1
        feats['start_month'] = min(obs)
        feats['end_month']   = max(obs)
        feats['mid_month']   = (min(obs)+max(obs))/2
    else:
        feats['window_size']=feats['start_month']=feats['end_month']=feats['mid_month']=0

    mv = {}
    for m in obs:
        v = compute_monthly_all(row, m)
        if v: mv[m] = v

    if not mv: return feats

    def arr(key):
        return np.array([mv[m][key] for m in mv if not np.isnan(mv[m].get(key, np.nan))])

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

    for key in ['mndwi','ndwi','ndvi','lswi','awei','sar','rvi','sar_diff','ndre','ndci','bsi']:
        pstats(arr(key), key)

    for tm in [0.0, 0.05, 0.1, 0.15]:
        for ts in [0.03, 0.05, 0.08]:
            cnt = sum(1 for m in mv if mv[m]['mndwi']>tm and mv[m]['sar']<ts)
            feats[f'wp_{int(tm*100)}_{int(ts*100)}']=cnt/len(mv)

    mndwi_a=arr('mndwi'); sar_a=arr('sar')
    if len(mndwi_a)>0 and len(sar_a)>0:
        feats['water_score']=feats['mndwi_p25']*(1-min(feats['sar_p25'],1))
        feats['water_floor']=feats['mndwi_p10']-feats['sar_p10']
        feats['water_consist']=feats['mndwi_frac_pos']*(1-feats['sar_frac_pos'])
        feats['min_mndwi_pos']=float(feats['mndwi_min']>0)
        feats['max_sar_low']=float(feats['sar_max']<0.1)
        valid_ms=list(mv.keys())
        if len(valid_ms)>=2:
            mvals=np.array([mv[m]['mndwi'] for m in valid_ms])
            svals=np.array([mv[m]['sar'] for m in valid_ms])
            corr=np.corrcoef(mvals,svals)[0,1] if np.std(mvals)>0 and np.std(svals)>0 else 0
            feats['mndwi_sar_corr']=corr
        else: feats['mndwi_sar_corr']=0
    else:
        for k in ['water_score','water_floor','water_consist','min_mndwi_pos','max_sar_low','mndwi_sar_corr']:
            feats[k]=np.nan
    return feats

# ---- Repeat simulated CV N_REPEATS times to reduce variance ----
N_REPEATS = 5
print(f"\nRunning {N_REPEATS} repeated simulated-test CVs...\n")

all_scores = []

for repeat in range(N_REPEATS):
    rng = np.random.RandomState(42 + repeat)
    
    # 80/20 stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42+repeat)
    y = train['label'].values
    tr_idx, val_idx = next(sss.split(np.zeros(len(train)), y))

    # Train features: full 12 months
    tr_rows = []
    for i in tr_idx:
        row = train.iloc[i]
        r = extract_features(row, list(range(1,13)))
        tr_rows.append(r)
    X_tr = pd.DataFrame(tr_rows)
    y_tr = train['label'].iloc[tr_idx].values

    # Validation features: simulated PARTIAL windows (matching test distribution)
    val_rows = []
    for i in val_idx:
        row = train.iloc[i]
        obs = sample_test_window()  # Draw from real test distribution
        r = extract_features(row, obs)
        val_rows.append(r)
    X_val = pd.DataFrame(val_rows)
    y_val = train['label'].iloc[val_idx].values

    feature_cols = [c for c in X_tr.columns]
    
    # Train LGB
    m_lgb = lgb.LGBMClassifier(random_state=42+repeat, n_estimators=600, learning_rate=0.03,
                                max_depth=6, scale_pos_weight=1.5, verbose=-1, min_child_samples=10)
    m_lgb.fit(X_tr[feature_cols], y_tr)
    p_lgb = m_lgb.predict_proba(X_val[feature_cols])[:, 1]

    # Train CB
    m_cb = CatBoostClassifier(random_seed=42+repeat, iterations=600, learning_rate=0.03,
                              depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0)
    m_cb.fit(X_tr[feature_cols], y_tr)
    p_cb = m_cb.predict_proba(X_val[feature_cols])[:, 1]

    p_ens = 0.5*p_lgb + 0.5*p_cb

    auc = roc_auc_score(y_val, p_ens)
    f1  = f1_score(y_val, p_ens >= 0.5)
    combined = 0.6*f1 + 0.4*auc
    all_scores.append({'repeat': repeat, 'auc': auc, 'f1': f1, 'combined': combined,
                       'n_pos_pred': (p_ens>=0.5).sum(), 'n_pos_true': y_val.sum()})
    print(f"  Repeat {repeat}: AUC={auc:.5f}, F1={f1:.5f}, Combined={combined:.5f} "
          f"(pred_pos={int((p_ens>=0.5).sum())}, true_pos={int(y_val.sum())})")

df_scores = pd.DataFrame(all_scores)
print(f"\n=== SIMULATED TEST CV (averaged over {N_REPEATS} repeats) ===")
print(f"  AUC:      {df_scores['auc'].mean():.5f} ± {df_scores['auc'].std():.5f}")
print(f"  F1:       {df_scores['f1'].mean():.5f} ± {df_scores['f1'].std():.5f}")
print(f"  Combined: {df_scores['combined'].mean():.5f} ± {df_scores['combined'].std():.5f}")
print(f"\nThis is the realistic local CV that correlates with the LB.")
print("Compare: LB actual scores were A=0.866, B=0.869")
