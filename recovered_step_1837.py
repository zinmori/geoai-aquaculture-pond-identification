    oof_final = w1*oof_lgb + w2*oof_cb + w3*oof_xgb
    iso_final = IsotonicRegression(out_of_bounds='clip')
    iso_final.fit(oof_final, y_tr_base)
    
    p_test_avg = (w1 * np.mean(test_preds_lgb, axis=0) +
                  w2 * np.mean(test_preds_cb, axis=0) +
                  w3 * np.mean(test_preds_xgb, axis=0))
    p_test_cal = iso_final.predict(p_test_avg)
    
    return oof_final, p_test_cal, p_test_avg

# ===== ITERATION 0: Base Model Training =====
print("\n===== Running Base Model (Iteration 0) =====")
oof_0, p_test_0, p_test_0_raw = train_ensemble_cv(X_train_base, y_train_base)
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
oof_1, p_test_1, p_test_1_raw = train_ensemble_cv(X_train_base, y_train_base, X_pseudo=X_pseudo, y_pseudo=y_pseudo)
pos_1 = (p_test_1 >= 0.5).sum()
print(f"Self-trained Model test set positives: {pos_1} / {len(p_test_1)}")

# ===== Generate Final Submission and Threshold-Tuned Variants =====
print("\nGenerating final submission files...")

# We map the probabilities for TargetRAUC so that they are strictly monotonic, 
# smooth, and satisfy the required formatting condition: (RAUC >= 0.5) == (F1 == 1)
def map_probabilities(raw_probs, binary_preds, cutoff_idx, K):
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

# 1. Default submission (threshold based on calibrated 0.5 probability)
sub_final_f1 = (p_test_1 >= 0.5).astype(int)
default_K = sub_final_f1.sum()
print(f"Default positive count: {default_K}")
default_mapped_probs = map_probabilities(p_test_1_raw, sub_final_f1, None, default_K)

sub_final = pd.DataFrame({
    'ID': test['ID'],
    'TargetF1': sub_final_f1,
    'TargetRAUC': default_mapped_probs
})
sub_final.to_csv('submission.csv', index=False)
sub_final.to_csv('submission_improved_final.csv', index=False)

# Assert format passes for default
assert ((sub_final['TargetRAUC'] >= 0.5) == (sub_final['TargetF1'] == 1)).all(), "F1/RAUC threshold mismatch!"
print(f"Saved default submission.csv with {default_K} positives.")

# 2. Generate threshold-tuned submissions targeting higher positive proportions
sorted_indices = np.argsort(p_test_1_raw)[::-1]

for K in [450, 480, 510, 540, 570, 600]:
    binary_preds = np.zeros(len(test), dtype=int)
    binary_preds[sorted_indices[:K]] = 1
    
    mapped_probs = map_probabilities(p_test_1_raw, binary_preds, None, K)
    
    sub_k = pd.DataFrame({
        'ID': test['ID'],
        'TargetF1': binary_preds,
        'TargetRAUC': mapped_probs
    })
    
    # Assert format checks pass
    assert ((sub_k['TargetRAUC'] >= 0.5) == (sub_k['TargetF1'] == 1)).all(), "F1/RAUC threshold mismatch!"
    
    fname = f'submission_top{K}.csv'
    sub_k.to_csv(fname, index=False)
    print(f"Saved {fname} with exactly {K} positive predictions.")

print("\nAll submission files generated and verified successfully!")