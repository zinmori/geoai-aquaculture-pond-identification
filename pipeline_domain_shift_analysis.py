"""
Analyse du domain shift géographique et test de features purement physiques.

Hypothèse: les 0.082 de gap entre simulated CV (0.951) et LB (0.869) viennent
du fait que les valeurs absolues de réflectance varient entre régions:
- Train region: X (ex: Vietnam)
- Test region: Y (ex: Bangladesh ou Chine) -> valeurs absolues différentes

Solution: utiliser des features qui ne dépendent PAS des valeurs absolues:
1. Ratios entre bandes (ex: MNDWI = (green-swir)/(green+swir) -> déjà normalisé)
2. Percentiles relatifs au propre pixel (ex: mndwi_month / mndwi_annual_mean)
3. Features binaires avec seuils physiques universels
4. Cross-band interactions comme VV/VH ratio

Test: comparer les distributions de features entre train et test
pour identifier lesquelles ont le plus de domain shift.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('Train.csv')
test  = pd.read_csv('Test.csv')

def compute_all_indices(row, obs):
    """Compute all indices for a set of observed months."""
    results = {'mndwi':[], 'ndwi':[], 'ndvi':[], 'lswi':[], 'awei':[],
               'sar':[], 'rvi':[], 'sar_diff':[], 'ndci':[], 'ndre':[],
               'blue':[], 'green':[], 'red':[], 'nir':[], 'swir1':[],
               'vh':[], 'vv':[], 'vh_lin':[], 'vv_lin':[]}
    for m in obs:
        ms = f'{m:02d}'
        blue=row[f'blue_{ms}']; green=row[f'green_{ms}']; red=row[f'red_{ms}']
        nir=row[f'nir_{ms}']; swir1=row[f'swir1_{ms}']; swir2=row[f'swir2_{ms}']
        vh=row[f'VH_{ms}']; vv=row[f'VV_{ms}']
        re1=row[f're1_{ms}']; re2=row[f're2_{ms}']
        if any(v==-9999 for v in [blue,green,red,nir,swir1,swir2,vh,vv]): continue
        vh_lin=10**(vh/10); vv_lin=10**(vv/10); sar=vh_lin+vv_lin
        results['mndwi'].append((green-swir1)/(green+swir1+1e-8))
        results['ndwi'].append((green-nir)/(green+nir+1e-8))
        results['ndvi'].append((nir-red)/(nir+red+1e-8))
        results['lswi'].append((nir-swir1)/(nir+swir1+1e-8))
        results['awei'].append(4*(green-swir1)-(0.25*nir+2.75*swir2))
        results['sar'].append(sar); results['rvi'].append(4*vh_lin/(sar+1e-8))
        results['sar_diff'].append(vv-vh)
        results['blue'].append(blue); results['green'].append(green)
        results['red'].append(red); results['nir'].append(nir)
        results['swir1'].append(swir1); results['vh'].append(vh); results['vv'].append(vv)
        results['vh_lin'].append(vh_lin); results['vv_lin'].append(vv_lin)
        if re1!=-9999 and re2!=-9999:
            results['ndci'].append((re1-red)/(re1+red+1e-8))
            results['ndre'].append((re2-re1)/(re2+re1+1e-8))
    return results

# Collect mean values per sample for train (label=0 and label=1) and test
print("Computing distributions...")
train_pond_stats = {k: [] for k in ['mndwi','ndwi','ndvi','sar','rvi','blue','nir','swir1','vh_lin']}
train_back_stats = {k: [] for k in ['mndwi','ndwi','ndvi','sar','rvi','blue','nir','swir1','vh_lin']}
test_stats       = {k: [] for k in ['mndwi','ndwi','ndvi','sar','rvi','blue','nir','swir1','vh_lin']}

for _, row in train.iterrows():
    r = compute_all_indices(row, range(1,13))
    label = row['label']
    for k in ['mndwi','ndwi','ndvi','sar','rvi','blue','nir','swir1','vh_lin']:
        if r[k]:
            val = np.mean(r[k])
            if label==1: train_pond_stats[k].append(val)
            else: train_back_stats[k].append(val)

for _, row in test.iterrows():
    obs = [m for m in range(1,13) if row[f'blue_{m:02d}']!=-9999]
    r = compute_all_indices(row, obs)
    for k in ['mndwi','ndwi','ndvi','sar','rvi','blue','nir','swir1','vh_lin']:
        if r[k]:
            test_stats[k].append(np.mean(r[k]))

print("\n=== Distribution Comparison (mean ± std) ===")
print(f"{'Feature':<12} {'Train Pond':>14} {'Train Back':>14} {'Test':>14} {'Train-Test Shift':>18}")
print("-" * 75)
for k in ['mndwi','ndwi','ndvi','sar','rvi','blue','nir','swir1','vh_lin']:
    tp = np.array(train_pond_stats[k]); tb = np.array(train_back_stats[k]); te = np.array(test_stats[k])
    tp_str = f"{tp.mean():.4f}±{tp.std():.4f}"
    tb_str = f"{tb.mean():.4f}±{tb.std():.4f}"
    te_str = f"{te.mean():.4f}±{te.std():.4f}"
    # Overall train vs test shift (using all train)
    all_train = np.concatenate([train_pond_stats[k], train_back_stats[k]])
    shift = abs(np.mean(all_train) - te.mean()) / (np.std(all_train) + 1e-8)
    print(f"{k:<12} {tp_str:>14} {tb_str:>14} {te_str:>14} {shift:>18.4f}")

print("\n=== Key insight: which features have largest train-test distribution shift? ===")
# Compute KL divergence proxy (bin-based) for each feature
print("Features sorted by shift magnitude (high shift = bad generalization):")
shifts = {}
for k in ['mndwi','ndwi','ndvi','sar','rvi','blue','nir','swir1','vh_lin']:
    all_train = np.array(train_pond_stats[k] + train_back_stats[k])
    te = np.array(test_stats[k])
    shift = abs(np.mean(all_train) - te.mean()) / (np.std(all_train) + 1e-8)
    shifts[k] = shift
for k, v in sorted(shifts.items(), key=lambda x: -x[1]):
    print(f"  {k:<12}: shift = {v:.4f}")

print("\n=== Separability of key features in TRAIN only (will compare to test quality) ===")
print("Pond vs Background separation (Cohen's d):")
for k in ['mndwi','ndwi','ndvi','sar','rvi','blue','nir','swir1']:
    tp = np.array(train_pond_stats[k]); tb = np.array(train_back_stats[k])
    pooled_std = np.sqrt((tp.std()**2 + tb.std()**2) / 2)
    d = (tp.mean() - tb.mean()) / (pooled_std + 1e-8)
    print(f"  {k:<12}: Cohen's d = {d:.4f} ({'STRONG' if abs(d)>1 else 'MODERATE' if abs(d)>0.5 else 'WEAK'})")

# Check if RATIO features are more stable
print("\n=== RATIO features (potentially more region-invariant) ===")
# VV/VH ratio in dB
vvvh_train_pond, vvvh_train_back, vvvh_test = [], [], []
for _, row in train.iterrows():
    r = compute_all_indices(row, range(1,13))
    vv = np.array(row[[f'VV_{m:02d}' for m in range(1,13)]].tolist())
    vh = np.array(row[[f'VH_{m:02d}' for m in range(1,13)]].tolist())
    valid = (vv != -9999) & (vh != -9999)
    if valid.any():
        ratio = np.mean((vv[valid] - vh[valid]))  # in dB
        if row['label']==1: vvvh_train_pond.append(ratio)
        else: vvvh_train_back.append(ratio)

for _, row in test.iterrows():
    obs = [m for m in range(1,13) if row[f'blue_{m:02d}']!=-9999]
    vv = np.array([row[f'VV_{m:02d}'] for m in obs])
    vh = np.array([row[f'VH_{m:02d}'] for m in obs])
    valid = (vv != -9999) & (vh != -9999)
    if valid.any():
        vvvh_test.append(np.mean((vv[valid] - vh[valid])))

tp = np.array(vvvh_train_pond); tb = np.array(vvvh_train_back); te = np.array(vvvh_test)
all_tr = np.concatenate([tp, tb])
shift = abs(np.mean(all_tr) - te.mean()) / (np.std(all_tr)+1e-8)
d = (tp.mean()-tb.mean())/(np.sqrt((tp.std()**2+tb.std()**2)/2)+1e-8)
print(f"  VV-VH ratio (dB): pond={tp.mean():.3f}, back={tb.mean():.3f}, test={te.mean():.3f}")
print(f"  Shift={shift:.4f}, Cohen's d={d:.4f}")
