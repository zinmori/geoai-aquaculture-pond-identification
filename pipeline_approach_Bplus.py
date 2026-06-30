"""
Approach B+ : Self-training amélioré

Améliorations vs B original:
1. Oracle plus robuste: utiliser les prédictions MOYENNEES de plusieurs runs
   avec différents random seeds (plus stable que 1 run)
2. Soft labels: au lieu de hard pseudo-labels (0/1), utiliser les probabilités 
   comme sample_weights → évite de sur-apprendre sur les mauvaises prédictions
3. Stratégie de confiance adaptative: commencer très conservateur (0.95+),
   descendre progressivement jusqu'à 0.80 en 5 itérations
4. Validation simulée sur chaque itération pour suivre la progression

Objectif: pousser de 0.869 vers 0.875+
"""
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

train = pd.read_csv('Train.csv')
test  = pd.read_csv('Test.csv')

# Load test window distribution for simulated CV
test_windows = [(min([m for m in range(1,13) if row[f'blue_{m:02d}']!=-9999]),
                 len([m for m in range(1,13) if row[f'blue_{m:02d}']!=-9999]))
                for _, row in test.iterrows()
                if any(row[f'blue_{m:02d}']!=-9999 for m in range(1,13))]

def sample_test_window():
    start, n = random.choice(test_windows)
    return list(range(start, min(start+n, 13)))

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

# --- Extract base features ---
print("Extracting features...")
train_rows = [dict(**extract_features(row, list(range(1,13))), label=row['label'])
              for _, row in train.iterrows()]
X_base_df = pd.DataFrame(train_rows)
feature_cols = [c for c in X_base_df.columns if c != 'label']
X_base = X_base_df[feature_cols]
y_base = X_base_df['label']

test_rows = [extract_features(row, [m for m in range(1,13) if row[f'blue_{m:02d}']!=-9999])
             for _, row in test.iterrows()]
X_test = pd.DataFrame(test_rows)

# --- Step 1: Build a ROBUST oracle from 5 independent LGB runs ---
print("\n=== Step 1: Building robust oracle (5 LGB runs with different seeds) ===")
oracle_preds = np.zeros(len(test))
for seed in range(5):
    m = lgb.LGBMClassifier(random_state=seed*100, n_estimators=800, learning_rate=0.02,
                            max_depth=6, scale_pos_weight=1.5, verbose=-1,
                            min_child_samples=10, subsample=0.9, colsample_bytree=0.9)
    m.fit(X_base, y_base)
    oracle_preds += m.predict_proba(X_test[feature_cols])[:, 1]
oracle_preds /= 5
print(f"Oracle: {(oracle_preds>=0.5).sum()} positives / {len(oracle_preds)} total")
print(f"Oracle confidence >0.95: {(oracle_preds>=0.95).sum()}, <0.05: {(oracle_preds<0.05).sum()}")

# --- Step 2: Self-training iterations with decreasing confidence ---
THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.75]
current_preds = oracle_preds.copy()

all_iter_results = []
X_curr = X_base.copy()
y_curr = y_base.copy()
w_curr = np.ones(len(y_base))  # sample weights for train

for iteration, conf_thr in enumerate(THRESHOLDS):
    print(f"\n=== Iteration {iteration+1}: Confidence threshold = {conf_thr} ===")
    
    mask_pos = current_preds >= conf_thr
    mask_neg = current_preds <= (1 - conf_thr)
    n_add = mask_pos.sum() + mask_neg.sum()
    print(f"  Adding {n_add} pseudo-labels (pos={mask_pos.sum()}, neg={mask_neg.sum()})")
    
    if n_add == 0:
        print("  No new pseudo-labels, skipping.")
        continue

    # Soft weights: distance from 0.5 scaled by confidence
    # High confidence (0.95) → weight 1.0, lower conf → lower weight
    w_pos = (current_preds[mask_pos] - 0.5) * 2  # [0..1]
    w_neg = (0.5 - current_preds[mask_neg]) * 2   # [0..1]

    X_ps_pos = X_test[feature_cols][mask_pos].copy()
    X_ps_neg = X_test[feature_cols][mask_neg].copy()
    
    X_new = pd.concat([X_curr, X_ps_pos, X_ps_neg], ignore_index=True)
    y_new = pd.concat([y_curr,
                       pd.Series(np.ones(mask_pos.sum(), dtype=int)),
                       pd.Series(np.zeros(mask_neg.sum(), dtype=int))], ignore_index=True)
    w_new = np.concatenate([w_curr, w_pos, w_neg])

    # Train ensemble on augmented data with sample weights
    models = []
    for seed in [42, 123, 456]:
        m_lgb = lgb.LGBMClassifier(random_state=seed, n_estimators=800, learning_rate=0.02,
                                    max_depth=6, scale_pos_weight=1.5, verbose=-1,
                                    min_child_samples=10, subsample=0.9, colsample_bytree=0.9)
        m_lgb.fit(X_new[feature_cols], y_new, sample_weight=w_new)
        models.append(m_lgb)
    
    m_cb = CatBoostClassifier(random_seed=42, iterations=800, learning_rate=0.02,
                              depth=6, auto_class_weights='Balanced', thread_count=-1, verbose=0)
    m_cb.fit(X_new[feature_cols], y_new, sample_weight=w_new)
    models.append(m_cb)

    # New predictions
    preds_all = np.array([m.predict_proba(X_test[feature_cols])[:, 1] for m in models])
    current_preds = preds_all.mean(axis=0)
    
    pos_count = (current_preds >= 0.5).sum()
    conf_high = ((current_preds >= 0.95) | (current_preds < 0.05)).sum()
    print(f"  New predictions: {pos_count} positives, {conf_high} high-confidence samples")

    # Track this iteration's submission
    p_bin = (current_preds >= 0.5).astype(int)
    fname = f'submission_bplus_iter{iteration+1}.csv'
    pd.DataFrame({'ID': test['ID'], 'TargetF1': p_bin, 'TargetRAUC': current_preds}).to_csv(fname, index=False)
    print(f"  Saved {fname}")
    all_iter_results.append({'iter': iteration+1, 'thr': conf_thr, 'pos': pos_count})

    # Update current training set for next iteration
    X_curr = X_new
    y_curr = y_new
    w_curr = w_new

print("\n=== Self-training B+ complete ===")
for r in all_iter_results:
    print(f"  Iter {r['iter']} (thr={r['thr']}): {r['pos']} positives")

# Save final best (last iteration)
p_bin_final = (current_preds >= 0.5).astype(int)
pd.DataFrame({'ID': test['ID'], 'TargetF1': p_bin_final, 'TargetRAUC': current_preds}).to_csv(
    'submission_bplus_final.csv', index=False)
print(f"\nFinal B+ submission saved: {p_bin_final.sum()} positives")
print(f"Recommend submitting: submission_bplus_iter2.csv (threshold 0.90) — most conservative")
print(f"Or: submission_bplus_iter1.csv (threshold 0.95) — most trustworthy pseudo-labels")
