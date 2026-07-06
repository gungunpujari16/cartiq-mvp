"""
Event -> ShopperSession state machine. Each incoming event is a snapshot of
browser state (TRD S4.1 event schema); this module folds a stream of events
for one session_id into the running ShopperSession row that scoring.py,
discounts.py and analytics.py all read from.
"""
from datetime import datetime

from sqlalchemy.orm import Session as DbSession

from app.models import Event, ShopperSession
from app.schemas import EventIn

_STAGE_RANK = {"None": 0, "Product Page": 0, "Cart": 1, "Checkout": 2, "Payment Gateway": 3}


def _time_of_day(ts: datetime) -> str:
    h = ts.hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 21:
        return "evening"
    return "night"


def _bump_stage(session: ShopperSession, stage: str) -> None:
    if _STAGE_RANK[stage] > _STAGE_RANK.get(session.abandonment_point, 0):
        session.abandonment_point = stage


def get_or_create_session(db: DbSession, brand_id: str, session_id: str, event: EventIn) -> ShopperSession:
    session = db.get(ShopperSession, session_id)
    if session is None:
        # SQLAlchemy column `default=` values only apply at INSERT/flush time,
        # not immediately on construction -- apply_event() runs right after
        # this and needs real numbers (e.g. `+=`) straight away, so set them
        # explicitly here instead of relying on the mapped defaults.
        session = ShopperSession(
            id=session_id,
            brand_id=brand_id,
            device_type=event.device_type.value,
            traffic_source=event.traffic_source or "direct",
            time_of_day=_time_of_day(event.timestamp),
            first_seen=event.timestamp,
            time_on_site=0.0,
            pages_viewed=0,
            cart_value=0.0,
            items_in_cart=0,
            scroll_depth_avg=0.0,
            exit_intent_count=0,
            payment_attempts=0,
            discount_used=False,
            added_to_cart=False,
            converted=False,
            abandonment_point="Product Page",
        )
        db.add(session)
    return session


def apply_event(db: DbSession, session: ShopperSession, event: EventIn) -> None:
    session.device_type = event.device_type.value
    session.last_seen = event.timestamp

    if event.traffic_source:
        session.traffic_source = event.traffic_source
    if event.return_customer is not None:
        session.return_customer = event.return_customer
    if event.product_category:
        session.product_category = event.product_category
    if event.cart_value is not None:
        session.cart_value = event.cart_value
    if event.cart_items is not None:
        session.items_in_cart = event.cart_items
    if event.scroll_depth is not None:
        # running average across events seen so far this session
        n = max(session.pages_viewed, 1)
        session.scroll_depth_avg = ((session.scroll_depth_avg * (n - 1)) + event.scroll_depth) / n
    if event.time_on_page is not None:
        session.time_on_site += event.time_on_page / 60000  # ms -> minutes

    if event.event_type.value == "page_view":
        session.pages_viewed += 1
        _bump_stage(session, "Product Page")
    elif event.event_type.value == "add_to_cart":
        session.added_to_cart = True
        _bump_stage(session, "Cart")
    elif event.event_type.value == "checkout_start" or event.event_type.value == "checkout_step":
        _bump_stage(session, "Checkout")
    elif event.event_type.value == "payment_attempt":
        session.payment_attempts += 1
        _bump_stage(session, "Payment Gateway")
    elif event.event_type.value == "exit_intent":
        session.exit_intent_count += 1
    elif event.event_type.value == "purchase":
        session.converted = True
        if event.order_value is not None:
            session.order_value = event.order_value
        # "None" here means "did not abandon", matching the historical dataset's
        # convention -- funnel() OR's on the converted flag directly, so this is
        # cosmetic for reporting, not load-bearing for the funnel counts.
        session.abandonment_point = "None"

    db.add(
        Event(
            brand_id=session.brand_id,
            session_id=session.id,
            event_type=event.event_type.value,
            page_url=event.page_url,
            client_timestamp=event.timestamp,
            cart_value=event.cart_value,
            cart_items=event.cart_items,
            scroll_depth=event.scroll_depth,
            time_on_page=event.time_on_page,
        )
    )
