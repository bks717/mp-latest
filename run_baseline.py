import sys
import sys
sys.path.insert(0, 'F:/mp-latest/disaster-ai-api')
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pymongo import MongoClient

# Load training data from MongoDB (same as train_global_mesh_model.py)
c = MongoClient('mongodb://127.0.0.1:27017', serverSelectionTimeoutMS=5000)
db = c.floodwatch
docs = list(db.global_mesh_training_data.find({}))

sequences = []
statics = []
labels = []
metadata_list = []

for doc in docs:
    try:
        seq = np.array(doc["sequence"], dtype=np.float32)
        sta = np.array(doc["static"], dtype=np.float32)
        label = int(doc["label"])
        sequences.append(seq)
        statics.append(sta)
        labels.append(label)
        metadata_list.append({
            k: doc.get(k) for k in (
                "cell_id", "lat", "lon", "region", "basin", "continent",
                "flood_prone", "held_out", "label", "discharge_ratio",
                "retrieved_at_utc"
            ) if k in doc
        })
    except Exception as exc:
        continue

n = len(sequences)
n_pos = int(sum(labels))
print(f"Loaded {n} labelled samples ({n_pos} positive)")

# Geographical split (same as train_global_mesh_model.py)
order = sorted(range(n), key=lambda i: (metadata_list[i].get("lon", 0.0), metadata_list[i].get("lat", 0.0)))
n_test = max(1, int(n * 0.2))
n_val = max(1, int(n * 0.2))
n_train = n - n_val - n_test

test_idx = set(order[:n_test])
val_idx = set(order[n_test:n_test + n_val])
train_idx = set(order[n_test + n_val:])

# Extract static features + discharge ratio for baseline
# Static features: elevation, slope, TWI, land cover, coastal flag, GDACS distance
# Plus discharge ratio as an additional feature
X_train = []
y_train = []
X_test = []
y_test = []

for i in sorted(train_idx):
    statics_i = statics[i]
    discharge_ratio = metadata_list[i].get("discharge_ratio", 1.0)
    # Features: elevation, slope, TWI, land_cover, coastal, gdacs_distance, discharge_ratio
    feat = [statics_i[0], statics_i[1], statics_i[2], statics_i[3], statics_i[4], statics_i[5], discharge_ratio]
    X_train.append(feat)
    y_train.append(labels[i])

for i in sorted(test_idx):
    statics_i = statics[i]
    discharge_ratio = metadata_list[i].get("discharge_ratio", 1.0)
    feat = [statics_i[0], statics_i[1], statics_i[2], statics_i[3], statics_i[4], statics_i[5], discharge_ratio]
    X_test.append(feat)
    y_test.append(labels[i])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"Train: {len(X_train)} samples ({sum(y_train)} positive)")
print(f"Test: {len(X_test)} samples ({sum(y_test)} positive)")

# Logistic regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

lr_pred_proba = lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred_proba)
print(f"\nLogistic Regression (static features + discharge ratio):")
print(f"  Test AUC: {lr_auc:.4f}")
print(f"  Test samples: {len(X_test)}, Positives: {sum(y_test)}")

# Discharge ratio only baseline
X_test_dr = X_test[:, [6]]  # Just discharge ratio
lr_dr = LogisticRegression(max_iter=1000, random_state=42)
lr_dr.fit(X_train[:, [6]], y_train)
lr_dr_pred = lr_dr.predict_proba(X_test_dr)[:, 1]
lr_dr_auc = roc_auc_score(y_test, lr_dr_pred)
print(f"  Discharge ratio only AUC: {lr_dr_auc:.4f}")

# Static features only (no discharge ratio)
X_test_static = X_test[:, :6]
lr_static = LogisticRegression(max_iter=1000, random_state=42)
lr_static.fit(X_train[:, :6], y_train)
lr_static_pred = lr_static.predict_proba(X_test_static)[:, 1]
lr_static_auc = roc_auc_score(y_test, lr_static_pred)
print(f"  Static features only AUC: {lr_static_auc:.4f}")