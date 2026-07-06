"""
Shared feature engineering — used by BOTH ml/train_model.py (offline training
from historical CSV) and app/scoring.py (live inference from a ShopperSession
row), so train-time and serve-time encoding can never drift apart.

TRD S3.1 specifies 15 raw features. This build trains on the subset that is
actually derivable from the historical dataset (IPBL/ecommerce_cleaned.csv):
time_on_site, pages_viewed, cart_value(proxy via items_in_cart+added_to_cart),
device_type, traffic_source, return_customer_flag, time_of_day, engagement_score,
avg_time_per_page, product_category, coupon_attempted.

scroll_depth_avg, exit_intent_count and payment_attempts ARE captured live by
the snippet/ingestion API (see models.ShopperSession) and are wired into this
same feature builder -- they just default to 0 for the historical rows that
never recorded them, so a future retrain on live demo-store traffic will pick
them up automatically without any code change here.
"""
from __future__ import annotations

# Fixed vocabularies so one-hot columns are stable between train and serve,
# regardless of what categories happen to appear in a given batch/session.
DEVICE_TYPES = ["desktop", "mobile", "tablet"]
TRAFFIC_SOURCES = ["direct", "email", "organic", "paid search", "referral", "social media"]
TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]
PRODUCT_CATEGORIES = [
    "apparel", "beauty", "books", "electronics",
    "grocery", "home & kitchen", "sports", "toys", "unknown",
]

# Passed straight through from the caller's session dict (ShopperSession columns
# at serving time, historical-row fields at training time -- see ml/historical_data.py).
NUMERIC_PASSTHROUGH_FEATURES = [
    "time_on_site",
    "pages_viewed",
    "items_in_cart",
    "cart_value",
    "added_to_cart",
    "return_customer",
    "discount_used",
    "scroll_depth_avg",
    "exit_intent_count",
    "payment_attempts",
]


def _one_hot(value: str, vocab: list[str], prefix: str) -> dict[str, float]:
    value = (value or "unknown").strip().lower()
    return {f"{prefix}_{v}": (1.0 if value == v else 0.0) for v in vocab}


def build_feature_vector(session: dict) -> dict[str, float]:
    """session: dict with keys matching models.ShopperSession attribute names."""
    features: dict[str, float] = {}

    for key in NUMERIC_PASSTHROUGH_FEATURES:
        val = session.get(key, 0) or 0
        features[key] = float(val)

    # Derived internally (not stored columns) so train-time and serve-time
    # values can never disagree regardless of what the caller passes in.
    time_on_site = float(session.get("time_on_site", 0) or 0)
    pages_viewed = float(session.get("pages_viewed", 0) or 0)
    features["engagement_score"] = time_on_site * pages_viewed
    features["avg_time_per_page"] = time_on_site / (pages_viewed + 1)

    features.update(_one_hot(session.get("device_type", ""), DEVICE_TYPES, "device"))
    features.update(_one_hot(session.get("traffic_source", ""), TRAFFIC_SOURCES, "traffic"))
    features.update(_one_hot(session.get("time_of_day", ""), TIME_OF_DAY, "time"))
    category = (session.get("product_category") or "unknown").strip().lower()
    if category not in PRODUCT_CATEGORIES:
        category = "unknown"
    features.update(_one_hot(category, PRODUCT_CATEGORIES, "category"))

    return features


def feature_names() -> list[str]:
    """Deterministic column order used for both training and inference."""
    sample = build_feature_vector({})
    return sorted(sample.keys())
