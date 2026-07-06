"""
API key auth: X-CartIQ-Key header -> brand_id.

TRD S5: "API keys hashed with bcrypt (cost factor 12)". A raw key is shown to
the brand exactly once (at creation) and only its bcrypt hash is stored,
matching that requirement. key_prefix is stored in the clear purely as a fast
DB index to narrow candidates before running the (deliberately slow) bcrypt
comparison against each one.
"""
import secrets

from fastapi import Depends, Header, HTTPException, status
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ApiKey, Brand

pwd_context = CryptContext(schemes=["bcrypt"], bcrypt__rounds=12)


def generate_api_key() -> tuple[str, str, str]:
    """Returns (raw_key, prefix, hash) for a brand-new API key."""
    raw_key = f"ciq_{secrets.token_urlsafe(32)}"
    prefix = raw_key[:12]
    return raw_key, prefix, pwd_context.hash(raw_key)


def get_current_brand(
    x_cartiq_key: str = Header(..., alias="X-CartIQ-Key"),
    db: Session = Depends(get_db),
) -> Brand:
    prefix = x_cartiq_key[:12]
    candidates = db.query(ApiKey).filter(ApiKey.key_prefix == prefix).all()
    for candidate in candidates:
        if pwd_context.verify(x_cartiq_key, candidate.key_hash):
            brand = db.get(Brand, candidate.brand_id)
            if brand is not None:
                return brand
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
