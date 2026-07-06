"""
Pydantic request/response models. EVENT_TYPES matches TRD S4.1's event_type
enum exactly, and Event.NO_PII_FIELDS documents the PRD Feature 1 guarantee
(no name/email/phone/address/payment fields ever accepted here).
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    page_view = "page_view"
    add_to_cart = "add_to_cart"
    remove_from_cart = "remove_from_cart"
    checkout_start = "checkout_start"
    checkout_step = "checkout_step"
    payment_attempt = "payment_attempt"
    purchase = "purchase"
    exit_intent = "exit_intent"


class DeviceType(str, Enum):
    mobile = "mobile"
    desktop = "desktop"
    tablet = "tablet"


class EventIn(BaseModel):
    session_id: str
    event_type: EventType
    timestamp: datetime
    device_type: DeviceType
    page_url: str
    traffic_source: str | None = None
    cart_value: float | None = Field(default=None, ge=0)
    cart_items: int | None = Field(default=None, ge=0)
    scroll_depth: float | None = Field(default=None, ge=0, le=1)
    time_on_page: int | None = Field(default=None, ge=0)  # ms
    return_customer: bool | None = None
    order_value: float | None = Field(default=None, ge=0)
    product_category: str | None = None
    discount_code: str | None = None

    @field_validator("page_url")
    @classmethod
    def strip_query_params(cls, v: str) -> str:
        # PRD Feature 1: "query parameters with potential PII are stripped by snippet" --
        # enforced again server-side in case a misbehaving integration forgets.
        return v.split("?")[0]


class EventBatchIn(BaseModel):
    events: list[EventIn]


class EventBatchOut(BaseModel):
    accepted: int
    session_id: str


class ScoreOut(BaseModel):
    session_id: str
    score: float
    segment: str
    top_features: list[dict]
    computed_at: datetime


class PredictIn(BaseModel):
    session_id: str


class DiscountRequestIn(BaseModel):
    session_id: str


class DiscountOut(BaseModel):
    issued: bool
    reason: str
    code: str | None = None
    pct: float | None = None
    expires_at: datetime | None = None
    is_holdout: bool = False


class BrandCreateIn(BaseModel):
    name: str
    discount_score_threshold: float = 40.0
    discount_min_cart_value: float = 0.0
    discount_max_pct: float = 15.0


class BrandCreateOut(BaseModel):
    brand_id: str
    api_key: str  # returned once, at creation time only -- never stored in plaintext


class OverviewOut(BaseModel):
    sessions: int
    cart_add_rate: float
    conversion_rate: float
    avg_order_value: float
    return_customer_rate: float
    discount_usage_rate: float


class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float


class SegmentOut(BaseModel):
    label: str
    size: int
    avg_engagement: float
    avg_cart_value: float
    avg_conversion_rate: float
