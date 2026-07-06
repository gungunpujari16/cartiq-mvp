"""
CartIQ Ingestion + Scoring + Discount API (TRD S4). One FastAPI service
covers what the TRD splits across an API-Gateway-fronted Ingestion service
and a separate Scoring service -- reasonable to merge at this scale, split
later if either becomes a separate bottleneck.
"""
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session as DbSession

from app import analytics, discounts, scoring
from app.auth import generate_api_key, get_current_brand
from app.config import settings
from app.db import get_db, init_db
from app.ingestion import apply_event, get_or_create_session
from app.models import Brand, Discount, Score, ShopperSession
from app.schemas import (
    BrandCreateIn,
    BrandCreateOut,
    DiscountOut,
    DiscountRequestIn,
    EventBatchIn,
    EventBatchOut,
    PredictIn,
    ScoreOut,
)

app = FastAPI(title="CartIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/v1/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/v1/me")
def whoami(brand: Brand = Depends(get_current_brand)):
    """Resolves the calling API key to its brand -- lets the dashboard/snippet
    discover brand_id and discount settings without the integrator hardcoding it."""
    return {
        "brand_id": brand.id,
        "name": brand.name,
        "discount_score_threshold": brand.discount_score_threshold,
        "discount_min_cart_value": brand.discount_min_cart_value,
        "discount_max_pct": brand.discount_max_pct,
    }


# ── Brand / API key provisioning (admin-only in a real deployment; open here for the demo) ──
@app.post("/v1/brands", response_model=BrandCreateOut)
def create_brand(payload: BrandCreateIn, db: DbSession = Depends(get_db)):
    brand = Brand(
        name=payload.name,
        discount_score_threshold=payload.discount_score_threshold,
        discount_min_cart_value=payload.discount_min_cart_value,
        discount_max_pct=payload.discount_max_pct,
    )
    db.add(brand)
    db.commit()
    db.refresh(brand)

    raw_key, prefix, key_hash = generate_api_key()
    from app.models import ApiKey

    db.add(ApiKey(brand_id=brand.id, key_hash=key_hash, key_prefix=prefix))
    db.commit()

    return BrandCreateOut(brand_id=brand.id, api_key=raw_key)


# ── Ingestion (TRD S4: POST /v1/events) ──
@app.post("/v1/events", response_model=EventBatchOut)
def ingest_events(
    payload: EventBatchIn,
    brand: Brand = Depends(get_current_brand),
    db: DbSession = Depends(get_db),
):
    if not payload.events:
        raise HTTPException(status_code=400, detail="No events provided")

    session_id = payload.events[0].session_id
    session = get_or_create_session(db, brand.id, session_id, payload.events[0])
    for event in payload.events:
        if event.session_id != session_id:
            raise HTTPException(status_code=400, detail="Batch must contain a single session_id")
        apply_event(db, session, event)

    db.commit()
    return EventBatchOut(accepted=len(payload.events), session_id=session_id)


def _get_session_or_404(db: DbSession, brand_id: str, session_id: str) -> ShopperSession:
    session = db.get(ShopperSession, session_id)
    if session is None or session.brand_id != brand_id:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ── Scoring (TRD S4: POST /v1/score/predict, GET /v1/score/{session_id}) ──
@app.post("/v1/score/predict", response_model=ScoreOut)
def predict_score(
    payload: PredictIn,
    brand: Brand = Depends(get_current_brand),
    db: DbSession = Depends(get_db),
):
    session = _get_session_or_404(db, brand.id, payload.session_id)
    session_attrs = {c.name: getattr(session, c.name) for c in ShopperSession.__table__.columns}
    score, segment, top_features = scoring.score_session(session_attrs)

    record = Score(brand_id=brand.id, session_id=session.id, score=score, segment=segment, top_features=top_features)
    db.add(record)
    db.commit()
    db.refresh(record)

    return ScoreOut(
        session_id=session.id, score=score, segment=segment,
        top_features=top_features, computed_at=record.created_at,
    )


@app.get("/v1/score/{session_id}", response_model=ScoreOut)
def get_score(session_id: str, brand: Brand = Depends(get_current_brand), db: DbSession = Depends(get_db)):
    _get_session_or_404(db, brand.id, session_id)
    record = (
        db.query(Score)
        .filter(Score.session_id == session_id, Score.brand_id == brand.id)
        .order_by(Score.created_at.desc())
        .first()
    )
    if record is None:
        raise HTTPException(status_code=404, detail="No score computed yet for this session")
    return ScoreOut(
        session_id=session_id, score=record.score, segment=record.segment,
        top_features=record.top_features, computed_at=record.created_at,
    )


# ── Discounts (TRD S4: POST /v1/discounts/generate) ──
@app.post("/v1/discounts/generate", response_model=DiscountOut)
def generate_discount(
    payload: DiscountRequestIn,
    brand: Brand = Depends(get_current_brand),
    db: DbSession = Depends(get_db),
):
    session = _get_session_or_404(db, brand.id, payload.session_id)
    latest_score = (
        db.query(Score)
        .filter(Score.session_id == session.id, Score.brand_id == brand.id)
        .order_by(Score.created_at.desc())
        .first()
    )
    if latest_score is None:
        raise HTTPException(status_code=400, detail="Score the session first via /v1/score/predict")

    result = discounts.evaluate_and_generate(db, brand, session, latest_score.score)
    return DiscountOut(**result)


# ── Dashboard-facing analytics (TRD S4: GET /v1/brands/{brand_id}/analytics/...) ──
@app.get("/v1/brands/{brand_id}/analytics/overview")
def analytics_overview(brand_id: str, brand: Brand = Depends(get_current_brand), db: DbSession = Depends(get_db)):
    if brand_id != brand.id:
        raise HTTPException(status_code=403, detail="API key does not match brand_id")
    return analytics.overview(db, brand_id)


@app.get("/v1/brands/{brand_id}/analytics/funnel")
def analytics_funnel(brand_id: str, brand: Brand = Depends(get_current_brand), db: DbSession = Depends(get_db)):
    if brand_id != brand.id:
        raise HTTPException(status_code=403, detail="API key does not match brand_id")
    return analytics.funnel(db, brand_id)


@app.get("/v1/brands/{brand_id}/analytics/channels")
def analytics_channels(brand_id: str, brand: Brand = Depends(get_current_brand), db: DbSession = Depends(get_db)):
    if brand_id != brand.id:
        raise HTTPException(status_code=403, detail="API key does not match brand_id")
    return analytics.channels(db, brand_id)


@app.get("/v1/brands/{brand_id}/analytics/revenue")
def analytics_revenue(brand_id: str, brand: Brand = Depends(get_current_brand), db: DbSession = Depends(get_db)):
    if brand_id != brand.id:
        raise HTTPException(status_code=403, detail="API key does not match brand_id")
    return analytics.revenue(db, brand_id)


@app.get("/v1/brands/{brand_id}/analytics/discounts")
def analytics_discounts(brand_id: str, brand: Brand = Depends(get_current_brand), db: DbSession = Depends(get_db)):
    if brand_id != brand.id:
        raise HTTPException(status_code=403, detail="API key does not match brand_id")
    return analytics.discounts_summary(db, brand_id)


@app.get("/v1/brands/{brand_id}/segments")
def brand_segments(brand_id: str, brand: Brand = Depends(get_current_brand), db: DbSession = Depends(get_db)):
    if brand_id != brand.id:
        raise HTTPException(status_code=403, detail="API key does not match brand_id")
    return analytics.segments(db, brand_id)


@app.get("/v1/brands/{brand_id}/sessions")
def list_sessions(
    brand_id: str, page: int = 1, page_size: int = 100,
    brand: Brand = Depends(get_current_brand), db: DbSession = Depends(get_db),
):
    if brand_id != brand.id:
        raise HTTPException(status_code=403, detail="API key does not match brand_id")
    query = (
        db.query(ShopperSession)
        .filter(ShopperSession.brand_id == brand_id)
        .order_by(ShopperSession.last_seen.desc())
    )
    total = query.count()
    rows = query.offset((page - 1) * page_size).limit(page_size).all()

    def _with_latest_score(row: ShopperSession) -> dict:
        data = {c.name: getattr(row, c.name) for c in ShopperSession.__table__.columns}
        latest = (
            db.query(Score)
            .filter(Score.session_id == row.id, Score.brand_id == brand_id)
            .order_by(Score.created_at.desc())
            .first()
        )
        data["score"] = latest.score if latest else None
        data["segment"] = latest.segment if latest else None
        return data

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "sessions": [_with_latest_score(r) for r in rows],
    }
