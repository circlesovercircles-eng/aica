# balance_dataset.py
# Handles class imbalance on preprocessed CIC-IDS2017 data
# Requirements: pip install imbalanced-learn pandas scikit-learn

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
from joblib import parallel_backend

warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────────────────────────────────────
#  SETTINGS  –  tune these!
# ────────────────────────────────────────────────
INPUT_FILE  = "dataset/preprocessed_traffic.csv"          # your file
OUTPUT_FILE = "dataset/balanced_dataset_smote_selective.csv"   # where to save

TARGET_COL  = "Label_encoded"                         # your target column name

# Which classes to oversample (encoded values)
# Recommended: only medium-sized minorities
CLASSES_TO_OVERSAMPLE = [1,5,6,7,11,12]               # Bot, Slowhttptest, slowloris, FTP, SSH, Brute Force

# Target size for each oversampled minority class
TARGET_MINORITY_SIZE = 20000                          # e.g. bring them up to ~20k each

# How many BENIGN samples to keep (undersampling majority)
BENIGN_TARGET_COUNT = 600000                          # ← tune: 300k–800k is common

RANDOM_STATE = 42

# ────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Original shape: {df.shape}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("Original class distribution:")
print(pd.Series(y).value_counts().sort_index())

# ────────────────────────────────────────────────
# Strategy: Selective SMOTE + Majority Undersampling
# ────────────────────────────────────────────────

# 1. Separate majority (BENIGN = 0) for undersampling
mask_benign = (y == 0)
X_benign = X[mask_benign]
y_benign = y[mask_benign]

if len(y_benign) > BENIGN_TARGET_COUNT:
    print(f"Downsampling BENIGN from {len(y_benign):,} → {BENIGN_TARGET_COUNT:,}")
    
    # Randomly select BENIGN_TARGET_COUNT rows
    sample_idx = np.random.choice(
        len(y_benign),
        size=BENIGN_TARGET_COUNT,
        replace=False
    )
    
    X_benign_down = X_benign.iloc[sample_idx].reset_index(drop=True)
    y_benign_down = y_benign.iloc[sample_idx].reset_index(drop=True)
else:
    X_benign_down = X_benign.reset_index(drop=True)
    y_benign_down = y_benign.reset_index(drop=True)

# 2. Prepare data for selective oversampling
X_min = X[~mask_benign]
y_min = y[~mask_benign]

# Create sampling strategy dictionary only for chosen classes
sampling_strategy = {}
for cls in CLASSES_TO_OVERSAMPLE:
    current_count = (y_min == cls).sum()
    if current_count > 0 and current_count < TARGET_MINORITY_SIZE:
        sampling_strategy[cls] = TARGET_MINORITY_SIZE
    else:
        print(f"Skipping class {cls} (count = {current_count})")

if sampling_strategy:
    print("SMOTE strategy:", sampling_strategy)
    
    with parallel_backend('loky', n_jobs=-1):   # or 'threading'
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE
        )
        X_min_res, y_min_res = smote.fit_resample(X_min, y_min)

# 3. Combine everything back
X_bal = pd.concat([pd.DataFrame(X_benign_down), pd.DataFrame(X_min_res)], ignore_index=True)
y_bal = pd.concat([pd.Series(y_benign_down), pd.Series(y_min_res)], ignore_index=True)

# Shuffle
from sklearn.utils import shuffle
X_bal, y_bal = shuffle(X_bal, y_bal, random_state=RANDOM_STATE)

# ────────────────────────────────────────────────
# Final balanced dataset
# ────────────────────────────────────────────────
balanced_df = X_bal.copy()
balanced_df[TARGET_COL] = y_bal

print("\nBalanced class distribution:")
print(pd.Series(y_bal).value_counts().sort_index())

print(f"\nBalanced shape: {balanced_df.shape}")

# Save
balanced_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved → {OUTPUT_FILE}")

print("\nDone.")