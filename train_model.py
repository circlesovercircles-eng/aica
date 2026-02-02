# train_models.py
# Script to train and evaluate Random Forest, Decision Tree, and CatBoost on balanced CIC-IDS2017 dataset
# Requirements: pip install pandas scikit-learn catboost
# Run after balancing.py → uses the balanced file

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for saving models

# ────────────────────────────────────────────────
# SETTINGS – adjust if needed
# ────────────────────────────────────────────────
INPUT_FILE = "dataset/balanced_dataset_smote_selective.csv"  # from balancing.py
TARGET_COL = "Label_encoded"  # your target
TEST_SIZE = 0.30  # split
RANDOM_STATE = 42

# Label mapping for readable reports (copy from earlier)
LABEL_MAPPING = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack - Brute Force",
    13: "Web Attack - Sql Injection",
    14: "Web Attack - XSS"
}

# ────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────
print("Loading balanced dataset...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Shape: {df.shape}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Quick distribution check
print("\nLoaded class distribution:")
print(pd.Series(y).value_counts().sort_index())

# ────────────────────────────────────────────────
# Train/Test Split (stratified)
# ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y  # important!
)
print(f"\nTrain shape: {X_train.shape} | Test shape: {X_test.shape}")

# ────────────────────────────────────────────────
# Scaling (optional but recommended for DT/RF; CatBoost doesn't need it)
# ────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ────────────────────────────────────────────────
# Enhanced evaluation function (now includes full test metrics & CM)
# ────────────────────────────────────────────────
def evaluate_model(model, model_name, use_scaled=True):
    print(f"\n=== Training & Evaluating {model_name} ===")
    
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test
    
    model.fit(X_tr, y_train)
    
    y_pred = model.predict(X_te)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    
    # Classification report with names
    target_names = [LABEL_MAPPING.get(i, f"Class {i}") for i in sorted(np.unique(y))]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"cm_{model_name.lower().replace(' ', '_')}.png")
    plt.close()  # close figure to free memory
    print(f"Saved CM plot → cm_{model_name.lower().replace(' ', '_')}.png")
    
    # Save model
    joblib.dump(model, f"{model_name.lower().replace(' ', '_')}_model.pkl")
    print(f"Saved model → {model_name.lower().replace(' ', '_')}_model.pkl")
    
    return model, y_pred

# ────────────────────────────────────────────────
# Models (uncomment the ones you want)
# ────────────────────────────────────────────────

# Random Forest (commented out as in your code)
'''
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,            # to prevent overfitting
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight='balanced'  # handles remaining imbalance
)
evaluate_model(rf_clf, "Random Forest")
'''

# XGBoost
xgb_clf = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
evaluate_model(xgb_clf, "XGBoost")

# LightGBM (commented out as in your code)
'''
lgb_clf = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=9,
    learning_rate=0.07,
    random_state=42,
    class_weight='balanced',
    verbosity=-1
)
evaluate_model(lgb_clf, "LightGBM")
'''

# Voting Ensemble (optional - uncomment if you want)
'''
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')),
        ('xgb', XGBClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ('lgb', lgb.LGBMClassifier(n_estimators=400, random_state=42, class_weight='balanced', verbosity=-1))
    ],
    voting='soft'
)
evaluate_model(voting_clf, "Voting Ensemble")
'''

print("\nAll models trained and evaluated!")