"""
Trains the real-time purchase-probability model (PRD Feature 2 / TRD S3.1).

Algorithm: XGBoost Gradient Boosting Classifier -- per TRD S3.1, chosen for
highest AUC on tabular event data, fast (<5ms) inference, and native handling
of missing features common in partial sessions.

Data: ml/data/ecommerce_cleaned.csv -- a copy of the same 1,400-session dataset
used by the Phase 0 analytics dashboard (IPBL/ecommerce_cleaned.csv), duplicated
into this repo so the model can be trained on deploy without depending on a
sibling folder that isn't part of this git repo. See app/features.py's module
docstring for exactly which of the TRD's 15 features this trains on vs. which
are captured live only (scroll_depth_avg, exit_intent_count, payment_attempts).

Run:  python ml/train_model.py     (from the backend/ directory)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.config import settings
from app.features import build_feature_vector
from ml.historical_data import iter_historical_sessions

CSV_PATH = BACKEND_DIR / "ml" / "data" / "ecommerce_cleaned.csv"
SCORE_THRESHOLDS = {"high": 70, "medium": 40, "low": 20}  # segment cut points, see PRD Feature 2
DECISION_THRESHOLD = 0.40  # TRD S3.1: "Precision > 0.60 at threshold 0.40"


def load_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    fields = [f for _, f in iter_historical_sessions(CSV_PATH)]
    feature_dicts = [build_feature_vector(s) for s in fields]
    X = pd.DataFrame(feature_dicts).reindex(sorted(feature_dicts[0].keys()), axis=1)
    y = pd.Series([1 if f["converted"] else 0 for f in fields])
    return X, y


def main() -> None:
    X, y = load_training_frame()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= DECISION_THRESHOLD).astype(int)

    metrics = {
        "auc_roc": round(float(roc_auc_score(y_test, proba)), 4),
        "precision_at_0.40": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall_at_0.40": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
    }
    print("Evaluation metrics:", json.dumps(metrics, indent=2))

    Path(settings.model_path).parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(settings.model_path)

    meta = {
        "feature_columns": list(X.columns),
        "score_thresholds": SCORE_THRESHOLDS,
        "decision_threshold": DECISION_THRESHOLD,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(CSV_PATH.name),
    }
    with open(settings.model_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to {settings.model_path}")
    print(f"Saved metadata to {settings.model_meta_path}")


if __name__ == "__main__":
    main()
