import pandas as pd

# ────────────────────────────────────────────────
# Load your merged file (use appropriate path)
# ────────────────────────────────────────────────
file_path = "dataset/all_traffic_merged.csv"          # ← change if needed


print("Loading data...")
df = pd.read_csv(file_path, low_memory=False)

print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}\n")

# ────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────
threshold = 0.90          # 90%
min_rows_for_check = 1000 # skip very small columns if any (optional)

# ────────────────────────────────────────────────
# Find columns where ≥ threshold % rows have the SAME value
# ────────────────────────────────────────────────
near_constant_cols = []

for col in df.columns:
    try:
        # Skip if column has too few non-null values
        if df[col].notna().sum() < min_rows_for_check:
            continue
            
        # Most memory-efficient way: value_counts(normalize=True)
        vc = df[col].value_counts(normalize=True, dropna=False)
        
        if len(vc) == 0:
            continue
            
        top_freq = vc.iloc[0]           # frequency of the most common value
        most_common_val = vc.index[0]
        
        if top_freq >= threshold:
            near_constant_cols.append((col, top_freq * 100, most_common_val, len(vc)))
            
    except Exception as e:
        print(f"Skipped {col}: {e}")
        continue

# ────────────────────────────────────────────────
# Show results sorted by how "constant" they are
# ────────────────────────────────────────────────
if near_constant_cols:
    print(f"\nColumns with ≥ {threshold*100}% the same value:\n")
    print(f"{'Column':<45} {'Dominant %':>12} {'Most common value':<20} {'Unique values':>12}")
    print("-" * 95)
    
    for col, pct, val, uniques in sorted(near_constant_cols, key=lambda x: -x[1]):
        val_str = str(val)[:18] + "..." if len(str(val)) > 18 else str(val)
        print(f"{col:<45} {pct:>12.2f}%   {val_str:<20} {uniques:>12}")
else:
    print(f"No columns found with ≥ {threshold*100}% identical values.")

# Optional: save to file

'''
if near_constant_cols:
    pd.DataFrame(near_constant_cols, columns=['Column', 'Dominant_%', 'Most_Common', 'Unique_Count'])\
      .sort_values('Dominant_%', ascending=False)\
      .to_csv("near_constant_columns_90pct.csv", index=False)
    print("\nSaved detailed list → near_constant_columns_90pct.csv")
'''