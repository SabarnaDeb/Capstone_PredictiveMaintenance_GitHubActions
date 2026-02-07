import os
import joblib
from datasets import load_dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATASET_REPO = "SabarnaDeb/Capstone_PredictiveMaintenance"
TARGET = "engine_condition"

ds = load_dataset(DATASET_REPO, data_files={"train": "train.csv", "test": "test.csv"})
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

train_df.columns = train_df.columns.str.strip().str.lower().str.replace(" ", "_")
test_df.columns  = test_df.columns.str.strip().str.lower().str.replace(" ", "_")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

model = AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
}

os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/model.joblib")

with open("artifacts/metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

print("Training complete. Metrics:", metrics)
