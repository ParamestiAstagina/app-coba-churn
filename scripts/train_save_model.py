"""
Script ini dijalankan SEKALI untuk melatih dan menyimpan model terbaik.
Web Streamlit tidak menjalankan training ulang.

Cara menjalankan dari folder project:
python scripts/train_save_model.py
"""

import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score

DATA_PATH = "data/BankChurners.csv"
MODEL_PATH = "models/catboost_bayes_best_model.cbm"
METADATA_PATH = "models/model_metadata.json"
TARGET_COL = "Attrition_Flag"

DROP_COLS = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

# Hasil terbaik dari notebook penelitian
BEST_PARAMS = {
    "iterations": 435,
    "learning_rate": 0.08545901744057517,
    "depth": 5,
    "l2_leaf_reg": 3.3903849118545817,
}
BEST_THRESHOLD = 0.37

os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
df[TARGET_COL] = df[TARGET_COL].map({
    "Existing Customer": 0,
    "Attrited Customer": 1,
})

cat_features = df.select_dtypes(include=["object"]).columns.tolist()
if TARGET_COL in cat_features:
    cat_features.remove(TARGET_COL)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

model = CatBoostClassifier(
    loss_function="Logloss",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=0,
    thread_count=4,
    **BEST_PARAMS,
)

model.fit(
    X_tr_final,
    y_tr_final,
    cat_features=cat_features,
    eval_set=(X_val_final, y_val_final),
    early_stopping_rounds=50,
    use_best_model=True,
)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= BEST_THRESHOLD).astype(int)

metrics = {
    "Accuracy": float(accuracy_score(y_test, pred)),
    "Precision": float(precision_score(y_test, pred)),
    "Recall": float(recall_score(y_test, pred)),
    "F1-Score": float(f1_score(y_test, pred)),
    "F2-Score": float(fbeta_score(y_test, pred, beta=2)),
    "ROC-AUC": float(roc_auc_score(y_test, prob)),
}

model.save_model(MODEL_PATH)

metadata = {
    "threshold": BEST_THRESHOLD,
    "best_params": BEST_PARAMS,
    "metrics": metrics,
    "cat_features": cat_features,
    "feature_columns": X.columns.tolist(),
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("Model berhasil disimpan:", MODEL_PATH)
print("Metadata berhasil disimpan:", METADATA_PATH)
print("Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
