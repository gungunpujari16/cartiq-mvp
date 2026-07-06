"""
Dynamic Discount Engine (PRD Feature 3).

Rule: score < brand.discount_score_threshold AND cart_value > brand's minimum
AND no discount already issued this session -> generate a coupon.
A random 10% holdout group is still recorded (is_holdout=True) but never shown
a code, so the dashboard's Discounts tab can measure incremental lift vs a
no-intervention control (TRD S3 dashboard spec, PRD Feature 3 "A/B Testing").

Frequency caps: PRD specifies "max 3 per customer per 30 days", but sessions
here are anonymous (no persistent customer_id from a brand login system), so
this MVP enforces "max 1 per session" only -- the customer-level cap is a
documented limitation, not a silent omission.
"""
import random
import secrets
from datetime import datetime, timedelta

from sqlalchemy.orm import Session as DbSession

from app.models import Brand, Discount, ShopperSession

HOLDOUT_RATE = 0.10
COUPON_TTL = timedelta(hours=2)


def _generate_code() -> str:
    return f"CIQ-{secrets.token_hex(4).upper()}"


def evaluate_and_generate(db: DbSession, brand: Brand, session: ShopperSession, score: float) -> dict:
    existing = (
        db.query(Discount)
        .filter(Discount.session_id == session.id, Discount.brand_id == brand.id)
        .first()
    )
    if existing is not None:
        return {"issued": False, "reason": "Discount already issued for this session"}

    if score >= brand.discount_score_threshold:
        return {"issued": False, "reason": "Score above discount threshold"}

    if session.cart_value <= brand.discount_min_cart_value:
        return {"issued": False, "reason": "Cart value below brand minimum"}

    is_holdout = random.random() < HOLDOUT_RATE

    pct = 10.0 + (5.0 if session.return_customer else 0.0)
    pct = min(pct, brand.discount_max_pct)

    discount = Discount(
        brand_id=brand.id,
        session_id=session.id,
        code="" if is_holdout else _generate_code(),
        pct=pct,
        is_holdout=is_holdout,
        expires_at=datetime.utcnow() + COUPON_TTL,
    )
    db.add(discount)
    session.discount_used = session.discount_used or not is_holdout
    db.commit()
    db.refresh(discount)

    if is_holdout:
        # Recorded for lift measurement, but the shopper sees no intervention.
        return {"issued": False, "reason": "Holdout group (control, no intervention shown)", "is_holdout": True}

    return {
        "issued": True,
        "reason": "Eligible: low score + cart value above brand minimum",
        "code": discount.code,
        "pct": discount.pct,
        "expires_at": discount.expires_at,
        "is_holdout": False,
    }
