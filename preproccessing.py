# preprocessing_script.py
# This script preprocesses the CIC-IDS2017 (or similar) merged dataset for an ML-based IDS model.
# Steps explained in comments:
# 1. Load the data.
# 2. Drop meaningless columns: 
#    - Pre-defined useless ones (constants like bulk rates, always 0 in dataset).
#    - Dynamically find and drop near-constant columns (≥90% same value).
#    - Also drop non-feature columns like 'Flow ID', 'Timestamp' if present (unique/irrelevant).
# 3. Handle inf/NaN: Replace inf with NaN, then drop rows with any NaN.
# 4. Separate X (features) and y (target 'Label').
# 5. Ensure y matches cleaned X (same rows after drops).
# 6. Label encode y (convert categorical labels to integers for ML).
# 7. Compute Mutual Information (MI) scores between each feature in X and y.
# 8. Select features where MI > 0.1 (threshold for meaningful contribution).
# 9. Visualize MI scores with a bar graph (sorted, saved as PNG).
# 10. Create final preprocessed DF: selected X + encoded y (as 'Label_encoded').
# 11. Save as CSV.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns  # For better visualization 
from joblib import parallel_backend

# Step 1: Load the merged data
# Change path if needed
file_path = "dataset/all_traffic_merged.csv"  # Or your cleaned merged file
df = pd.read_csv(file_path, low_memory=False)
print(f"Original shape: {df.shape}")

# Clean column names (just in case, from previous steps)
df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9_.-]', '_', regex=True)

# Step 2: Drop meaningless columns
# 2a: Pre-defined useless/constant columns (from earlier analysis)
useless_cols = [
    'Fwd_Avg_Bytes_Bulk', 'Fwd_Avg_Packets_Bulk', 'Fwd_Avg_Bulk_Rate',
    'Bwd_Avg_Bytes_Bulk', 'Bwd_Avg_Packets_Bulk', 'Bwd_Avg_Bulk_Rate',
    'Bwd_PSH_Flags', 'Fwd_URG_Flags', 'Bwd_URG_Flags',
    'CWE_Flag_Count', 'ECE_Flag_Count', 'RST_Flag_Count'
]

# 2b: Dynamically find near-constant columns (≥90% same value)
threshold = 0.90
near_constant = []
for col in df.columns:
    if col == 'Label':  # Skip target
        continue
    vc = df[col].value_counts(normalize=True, dropna=False)
    if len(vc) > 0 and vc.iloc[0] >= threshold:
        near_constant.append(col)

# 2c: Drop non-feature columns if present (unique identifiers, not useful for ML)
non_features = ['Flow_ID', 'Timestamp']  # Common in CIC-IDS2017

# Combine all to drop
cols_to_drop = list(set(useless_cols + near_constant + non_features))
cols_to_drop = [col for col in cols_to_drop if col in df.columns]  # Ensure they exist

df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} meaningless columns: {cols_to_drop}")
print(f"Shape after drop: {df.shape}")

# Step 3: Handle inf/NaN
# Replace inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with any NaN
df.dropna(inplace=True)
print(f"Shape after dropping NaN/inf rows: {df.shape}")

# Step 4: Separate X and y
# Assume 'Label' is the target column (change if different)
if 'Label' not in df.columns:
    raise ValueError("Target column 'Label' not found!")
X = df.drop('Label', axis=1)
y = df['Label'].copy()  # y_full matches the cleaned rows now

# Step 5: y already matches cleaned X (since dropped from df before separating)

# Step 6: Label encode y (and all data if any categorical in X, but X is usually numerical)
# First, encode y (categorical to int)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Encoded labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# If X has any object/string columns, encode them too (though rare in this dataset)
cat_cols = X.select_dtypes(include=['object']).columns
if len(cat_cols) > 0:
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col])
    print(f"Encoded categorical columns in X: {cat_cols}")

# Step 7: Calculate MI scores
# MI requires numerical X and encoded y
# Sample if too large (MI is computationally intensive; optional for big data)
# X_sample = X.sample(100000)  # Uncomment if memory issues; adjust size
# y_sample = y_encoded[X_sample.index]

with parallel_backend('loky', inner_max_num_threads=1):
    mi_scores = mutual_info_classif(
        X, y,
        random_state=42,
        n_neighbors=2,           # ← key speedup: 2 instead of 3
        n_jobs=-1
    )

mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\nMI Scores (top 10):")
print(mi_series.head(10))

# Step 8: Select features with MI > 0.1
selected_features = mi_series[mi_series > 0.1].index.tolist()
X_selected = X[selected_features]
print(f"Selected {len(selected_features)} features with MI > 0.1: {selected_features}")

# Step 9: Visualize MI with graph
plt.figure(figsize=(12, 8))
sns.barplot(x=mi_series.values, y=mi_series.index, orient='h')
plt.title('Mutual Information Scores for Features')
plt.xlabel('MI Score')
plt.ylabel('Features')
# Highlight threshold
plt.axvline(x=0.1, color='r', linestyle='--', label='MI > 0.1 Threshold')
plt.legend()
plt.tight_layout()
plt.savefig('mi_scores_graph.png')
print("MI visualization saved as 'mi_scores_graph.png'")

# Step 10: Create final preprocessed DF
final_df = X_selected.copy()
final_df['Label_encoded'] = y_encoded  # Add encoded y

# Step 11: Save preprocessed dataset
output_path = "dataset/preprocessed_traffic.csv"
final_df.to_csv(output_path, index=False)
print(f"Preprocessed data saved to: {output_path}")
print(f"Final shape: {final_df.shape}")