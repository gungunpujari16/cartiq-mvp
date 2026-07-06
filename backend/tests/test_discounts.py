"""
Discount rule engine (PRD Feature 3): score threshold, cart-value minimum,
one-per-session cap, and the 10% holdout control group.
"""
import random

from app import discounts


def test_eligible_session_gets_a_discount(db_session, brand, shopper_session):
    random.seed(1)  # deterministic: not in the 10% holdout at this seed
    result = discounts.evaluate_and_generate(db_session, brand, shopper_session, score=20.0)

    assert result["issued"] is True
    assert result["code"].startswith("CIQ-")
    assert 0 < result["pct"] <= brand.discount_max_pct


def test_score_above_threshold_is_not_eligible(db_session, brand, shopper_session):
    result = discounts.evaluate_and_generate(db_session, brand, shopper_session, score=75.0)

    assert result["issued"] is False
    assert "threshold" in result["reason"].lower()


def test_cart_value_below_minimum_is_not_eligible(db_session, brand, shopper_session):
    shopper_session.cart_value = 10.0  # brand minimum is 50.0
    result = discounts.evaluate_and_generate(db_session, brand, shopper_session, score=20.0)

    assert result["issued"] is False
    assert "cart value" in result["reason"].lower()


def test_only_one_discount_per_session(db_session, brand, shopper_session, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.99)  # force out of holdout
    first = discounts.evaluate_and_generate(db_session, brand, shopper_session, score=20.0)
    second = discounts.evaluate_and_generate(db_session, brand, shopper_session, score=20.0)

    assert first["issued"] is True
    assert second["issued"] is False
    assert "already issued" in second["reason"].lower()


def test_holdout_group_is_recorded_but_shown_no_code(db_session, brand, shopper_session, monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)  # force into the 10% holdout

    result = discounts.evaluate_and_generate(db_session, brand, shopper_session, score=20.0)

    assert result["issued"] is False
    assert result["is_holdout"] is True
    assert "code" not in result or result.get("code") is None
