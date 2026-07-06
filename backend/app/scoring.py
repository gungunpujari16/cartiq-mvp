"""
Real-time purchase-probability scoring (PRD Feature 2 / TRD S3.1).

Loads the XGBoost model trained by ml/train_model.py once at process start
and serves predictions in-process -- the lightweight equivalent of TRD's
"ONNX-exported model in FastAPI; loaded into memory at startup" serving spec,
minus the ONNX export step (unnecessary at this scale/model size).
"""
import json
import threading

import pandas as pd
import xgboost as xgb

from app.config import settings
from app.features import build_feature_vector

_lock = threading.Lock()
_booster: xgb.Booster | None = None
_meta: dict | None = None


def _load() -> tuple[xgb.Booster, dict]:
    global _booster, _meta
    if _booster is None:
        with _lock:
            if _booster is None:
                with open(settings.model_meta_path) as f:
                    meta = json.load(f)
                booster = xgb.Booster()
                booster.load_model(settings.model_path)
                booster.feature_names = meta["feature_columns"]
                _booster, _meta = booster, meta
    return _booster, _meta


def segment_for_score(score: float, thresholds: dict) -> str:
    if score >= thresholds["high"]:
        return "High Intent"
    if score >= thresholds["medium"]:
        return "Medium"
    if score >= thresholds["low"]:
        return "Low"
    return "Bounce Risk"


def score_session(session_attrs: dict) -> tuple[float, str, list[dict]]:
    """session_attrs: dict of ShopperSession field values (see models.py)."""
    booster, meta = _load()
    columns = meta["feature_columns"]

    raw_features = build_feature_vector(session_attrs)
    row = {col: raw_features.get(col, 0.0) for col in columns}
    frame = pd.DataFrame([row], columns=columns)
    dmatrix = xgb.DMatrix(frame, feature_names=columns)

    proba = float(booster.predict(dmatrix)[0])
    score = round(proba * 100, 1)
    segment = segment_for_score(score, meta["score_thresholds"])

    contribs = booster.predict(dmatrix, pred_contribs=True)[0]  # last entry is the bias term
    feature_contribs = list(zip(columns, contribs[:-1]))
    feature_contribs.sort(key=lambda pair: abs(pair[1]), reverse=True)
    top_features = [
        {"feature": name, "contribution": round(float(value), 4)}
        for name, value in feature_contribs[:3]
    ]

    return score, segment, top_features
