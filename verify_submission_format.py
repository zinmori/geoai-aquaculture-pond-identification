import pandas as pd

sub = pd.read_csv('submission.csv')
test = pd.read_csv('Test.csv')

print("Submission shape:", sub.shape)
print("Test shape:", test.shape)

# Check shape equivalence
assert sub.shape[0] == test.shape[0], "Row counts do not match!"
assert sub.shape[1] == 3, "Submission should have exactly 3 columns (ID, TargetF1, TargetRAUC)!"

# Check column names
assert list(sub.columns) == ['ID', 'TargetF1', 'TargetRAUC'], "Column names are incorrect!"

# Check ID equivalence in order
assert (sub['ID'] == test['ID']).all(), "IDs or row ordering do not match!"

# Check values are clean
assert not sub.isnull().any().any(), "There are missing values in the submission!"
assert sub['TargetF1'].isin([0, 1]).all(), "TargetF1 has values other than 0 or 1!"
assert ((sub['TargetRAUC'] >= 0.5) == (sub['TargetF1'] == 1)).all(), "Threshold classification mismatch!"

print("\n--- NEW SUBMISSION FORMAT CHECKS PASSED SUCCESSFULLY! ---")
