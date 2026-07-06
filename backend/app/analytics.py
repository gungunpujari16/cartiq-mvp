"""
Aggregation queries backing the dashboard's Overview / Funnel / Segments tabs
(PRD Feature 4). Computed on demand against Postgres/SQLite -- at MVP data
volumes this replaces TRD's separate Snowflake analytics warehouse.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func
from sqlalchemy.orm import Session as DbSession

from app.models import Discount, ShopperSession

FUNNEL_STAGES = ["Visited", "Added to Cart", "Reached Checkout", "Reached Payment", "Converted"]


def _brand_sessions(db: DbSession, brand_id: str):
    return db.query(ShopperSession).filter(ShopperSession.brand_id == brand_id)


def overview(db: DbSession, brand_id: str) -> dict:
    q = _brand_sessions(db, brand_id)
    total = q.count()
    if total == 0:
        return {
            "sessions": 0, "cart_add_rate": 0.0, "conversion_rate": 0.0,
            "avg_order_value": 0.0, "return_customer_rate": 0.0, "discount_usage_rate": 0.0,
        }

    cart_adds = q.filter(ShopperSession.added_to_cart.is_(True)).count()
    converted = q.filter(ShopperSession.converted.is_(True)).count()
    returners = q.filter(ShopperSession.return_customer.is_(True)).count()
    discounted = q.filter(ShopperSession.discount_used.is_(True)).count()
    avg_order_value = (
        db.query(func.avg(ShopperSession.order_value))
        .filter(ShopperSession.brand_id == brand_id, ShopperSession.converted.is_(True))
        .scalar()
        or 0.0
    )

    return {
        "sessions": total,
        "cart_add_rate": round(100 * cart_adds / total, 1),
        "conversion_rate": round(100 * converted / total, 1),
        "avg_order_value": round(float(avg_order_value), 2),
        "return_customer_rate": round(100 * returners / total, 1),
        "discount_usage_rate": round(100 * discounted / total, 1),
    }


def funnel(db: DbSession, brand_id: str) -> list[dict]:
    q = _brand_sessions(db, brand_id)
    total = q.count()
    if total == 0:
        return [{"stage": s, "count": 0, "drop_off_pct": 0.0} for s in FUNNEL_STAGES]

    reached_checkout_points = ("Checkout", "Payment Gateway")
    counts = {
        "Visited": total,
        "Added to Cart": q.filter(ShopperSession.added_to_cart.is_(True)).count(),
        "Reached Checkout": q.filter(
            (ShopperSession.abandonment_point.in_(reached_checkout_points))
            | (ShopperSession.converted.is_(True))
        ).count(),
        "Reached Payment": q.filter(
            (ShopperSession.abandonment_point == "Payment Gateway")
            | (ShopperSession.converted.is_(True))
        ).count(),
        "Converted": q.filter(ShopperSession.converted.is_(True)).count(),
    }

    stages = []
    prev = total
    for name in FUNNEL_STAGES:
        count = counts[name]
        drop_off = round(100 * (1 - count / prev), 1) if prev else 0.0
        stages.append({"stage": name, "count": count, "drop_off_pct": drop_off})
        prev = count if count else prev
    return stages


def channels(db: DbSession, brand_id: str) -> list[dict]:
    # Grouped in pandas rather than SQL so the aggregation stays portable
    # across both SQLite (local) and Postgres (production) without
    # dialect-specific boolean-cast syntax.
    df = pd.read_sql(_brand_sessions(db, brand_id).statement, db.bind)
    if df.empty:
        return []
    grouped = df.groupby("traffic_source").agg(
        sessions=("id", "count"),
        conversions=("converted", "sum"),
    ).reset_index()
    grouped["conversion_rate"] = (100 * grouped["conversions"] / grouped["sessions"]).round(1)
    return grouped.to_dict(orient="records")


def revenue(db: DbSession, brand_id: str) -> dict:
    df = pd.read_sql(_brand_sessions(db, brand_id).statement, db.bind)
    if df.empty:
        return {"by_category": [], "by_payment": [], "order_values": []}
    converted = df[df["converted"] == True]  # noqa: E712
    by_category = (
        converted.groupby("product_category")["order_value"].agg(["mean", "count"]).reset_index()
        if not converted.empty else pd.DataFrame(columns=["product_category", "mean", "count"])
    )
    return {
        "by_category": by_category.rename(columns={"mean": "avg_order_value", "count": "orders"}).to_dict(orient="records"),
        "order_values": converted["order_value"].dropna().tolist(),
    }


def discounts_summary(db: DbSession, brand_id: str) -> dict:
    df = pd.read_sql(
        db.query(Discount).filter(Discount.brand_id == brand_id).statement, db.bind
    )
    if df.empty:
        return {"issued": 0, "holdout": 0, "treated_conversion_rate": 0.0, "holdout_conversion_rate": 0.0}

    sessions = pd.read_sql(_brand_sessions(db, brand_id).statement, db.bind)
    merged = df.merge(sessions[["id", "converted"]], left_on="session_id", right_on="id", how="left")

    treated = merged[~merged["is_holdout"]]
    holdout = merged[merged["is_holdout"]]
    return {
        "issued": int((~merged["is_holdout"]).sum()),
        "holdout": int(merged["is_holdout"].sum()),
        "treated_conversion_rate": round(100 * treated["converted"].mean(), 1) if len(treated) else 0.0,
        "holdout_conversion_rate": round(100 * holdout["converted"].mean(), 1) if len(holdout) else 0.0,
    }


def segments(db: DbSession, brand_id: str, k: int = 3) -> dict:
    """Behavioral customer segments (PRD Feature 4 'Segments' tab) -- distinct
    from the BRD's brand-tier WTP segmentation, which belongs to the separate
    B2B survey analytics and is out of scope here."""
    df = pd.read_sql(_brand_sessions(db, brand_id).statement, db.bind)
    if len(df) < k:
        return {"profiles": [], "points": []}

    features = df[["time_on_site", "pages_viewed", "cart_value"]].fillna(0)
    features["engagement_score"] = df["time_on_site"] * df["pages_viewed"]
    scaled = StandardScaler().fit_transform(features)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    df = df.assign(cluster=clusters, pca_x=coords[:, 0], pca_y=coords[:, 1])

    profile_rows = (
        df.groupby("cluster")
        .agg(
            size=("id", "count"),
            avg_engagement=("time_on_site", lambda s: round(float((s * df.loc[s.index, "pages_viewed"]).mean()), 1)),
            avg_cart_value=("cart_value", lambda s: round(float(s.mean()), 2)),
            avg_conversion_rate=("converted", lambda s: round(100 * float(s.mean()), 1)),
        )
        .reset_index()
    )
    # Label clusters by engagement so labels are stable across runs regardless of arbitrary cluster ids.
    ordered = profile_rows.sort_values("avg_engagement")
    labels = ["At-Risk", "Engaged", "Champions"][: len(ordered)]
    label_map = dict(zip(ordered["cluster"], labels))
    profile_rows["label"] = profile_rows["cluster"].map(label_map)

    profiles = profile_rows.drop(columns=["cluster"]).to_dict(orient="records")
    points = [
        {"session_id": row.id, "cluster": label_map[row.cluster], "pca_x": round(float(row.pca_x), 3), "pca_y": round(float(row.pca_y), 3)}
        for row in df.itertuples()
    ]
    return {"profiles": profiles, "points": points}
