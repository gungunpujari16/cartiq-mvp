"""
ORM models. Every table that holds tenant data carries brand_id so every
query can enforce tenant isolation at the application layer (see auth.py
and analytics.py) -- the lightweight equivalent of TRD S2.1's per-tenant
partition keys / row-level security.
"""
import uuid
from datetime import datetime, timedelta

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


class Brand(Base):
    __tablename__ = "brands"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Discount Engine settings (PRD Feature 3: brand-configurable margin protection)
    discount_score_threshold: Mapped[float] = mapped_column(Float, default=40.0)
    discount_min_cart_value: Mapped[float] = mapped_column(Float, default=0.0)
    discount_max_pct: Mapped[float] = mapped_column(Float, default=15.0)

    api_keys: Mapped[list["ApiKey"]] = relationship(back_populates="brand")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    brand_id: Mapped[str] = mapped_column(ForeignKey("brands.id"), index=True)
    key_hash: Mapped[str] = mapped_column(String, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String, index=True)  # fast lookup, full key verified via hash
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    brand: Mapped["Brand"] = relationship(back_populates="api_keys")


class ShopperSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # session_id from the snippet
    brand_id: Mapped[str] = mapped_column(ForeignKey("brands.id"), index=True)

    device_type: Mapped[str] = mapped_column(String, default="desktop")
    traffic_source: Mapped[str] = mapped_column(String, default="direct")
    time_of_day: Mapped[str] = mapped_column(String, default="morning")
    return_customer: Mapped[bool] = mapped_column(Boolean, default=False)
    product_category: Mapped[str | None] = mapped_column(String, nullable=True)

    time_on_site: Mapped[float] = mapped_column(Float, default=0.0)       # seconds
    pages_viewed: Mapped[int] = mapped_column(Integer, default=0)
    cart_value: Mapped[float] = mapped_column(Float, default=0.0)
    items_in_cart: Mapped[int] = mapped_column(Integer, default=0)
    scroll_depth_avg: Mapped[float] = mapped_column(Float, default=0.0)
    exit_intent_count: Mapped[int] = mapped_column(Integer, default=0)
    payment_attempts: Mapped[int] = mapped_column(Integer, default=0)
    discount_used: Mapped[bool] = mapped_column(Boolean, default=False)

    added_to_cart: Mapped[bool] = mapped_column(Boolean, default=False)
    converted: Mapped[bool] = mapped_column(Boolean, default=False)
    order_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    abandonment_point: Mapped[str] = mapped_column(String, default="None")

    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_seed: Mapped[bool] = mapped_column(Boolean, default=False)  # replayed historical row vs live browser session


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    brand_id: Mapped[str] = mapped_column(ForeignKey("brands.id"), index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)

    event_type: Mapped[str] = mapped_column(String, nullable=False)
    page_url: Mapped[str] = mapped_column(String, default="/")
    client_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    server_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    cart_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    cart_items: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scroll_depth: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_on_page: Mapped[int | None] = mapped_column(Integer, nullable=True)  # ms

    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Score(Base):
    __tablename__ = "scores"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    brand_id: Mapped[str] = mapped_column(ForeignKey("brands.id"), index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)

    score: Mapped[float] = mapped_column(Float)          # 0-100
    segment: Mapped[str] = mapped_column(String)          # High Intent / Medium / Low / Bounce Risk
    top_features: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Discount(Base):
    __tablename__ = "discounts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    brand_id: Mapped[str] = mapped_column(ForeignKey("brands.id"), index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)

    code: Mapped[str] = mapped_column(String)
    pct: Mapped[float] = mapped_column(Float)
    is_holdout: Mapped[bool] = mapped_column(Boolean, default=False)  # in holdout: no discount shown, used to measure lift
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.utcnow() + timedelta(hours=2)
    )
    redeemed: Mapped[bool] = mapped_column(Boolean, default=False)
