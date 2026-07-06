"""
Event schema validation (TRD S4.1): valid event_type enum, query-string
stripping (PRD Feature 1: no PII in page URLs), and rejection of malformed
payloads before they ever reach the DB.
"""
import pytest
from pydantic import ValidationError

from app.schemas import EventIn


def _base_event(**overrides):
    payload = {
        "session_id": "sess_1",
        "event_type": "page_view",
        "timestamp": "2026-01-01T00:00:00Z",
        "device_type": "mobile",
        "page_url": "/product/1",
    }
    payload.update(overrides)
    return payload


def test_valid_event_parses():
    event = EventIn(**_base_event())
    assert event.event_type.value == "page_view"


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValidationError):
        EventIn(**_base_event(event_type="add_credit_card"))


def test_unknown_device_type_is_rejected():
    with pytest.raises(ValidationError):
        EventIn(**_base_event(device_type="smart_fridge"))


def test_query_params_are_stripped_from_page_url():
    event = EventIn(**_base_event(page_url="/checkout?email=someone@example.com&token=abc"))
    assert event.page_url == "/checkout"
    assert "email" not in event.page_url


def test_negative_cart_value_is_rejected():
    with pytest.raises(ValidationError):
        EventIn(**_base_event(cart_value=-10))


def test_scroll_depth_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        EventIn(**_base_event(scroll_depth=1.5))
