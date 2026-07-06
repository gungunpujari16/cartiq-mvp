"""
Creates a demo brand + API key and replays the historical dataset
(IPBL/ecommerce_cleaned.csv) into the live DB as ShopperSession rows, each
scored by the trained model, so the dashboard has realistic data to show
immediately -- without waiting on real demo-store traffic to build up.

Idempotent: re-running it reuses the existing "CartIQ Demo Store" brand
instead of creating a duplicate, but re-seeds sessions each time (drops
previously seeded rows first) so this is safe to re-run after retraining.

Run:  python ml/seed_demo_data.py     (from the backend/ directory)
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app import scoring
from app.auth import generate_api_key
from app.db import SessionLocal, init_db
from app.models import ApiKey, Brand, Score, ShopperSession
from ml.historical_data import iter_historical_sessions

CSV_PATH = BACKEND_DIR.parent.parent / "IPBL" / "ecommerce_cleaned.csv"
DEMO_BRAND_NAME = "CartIQ Demo Store"
KEY_FILE = BACKEND_DIR / "demo_api_key.txt"


def get_or_create_demo_brand(db) -> tuple[Brand, str]:
    brand = db.query(Brand).filter(Brand.name == DEMO_BRAND_NAME).first()
    if brand is not None:
        if KEY_FILE.exists():
            return brand, KEY_FILE.read_text().strip()
        # Brand exists but we lost the raw key (it's only ever stored hashed) -- issue a new one.
        raw_key, prefix, key_hash = generate_api_key()
        db.add(ApiKey(brand_id=brand.id, key_hash=key_hash, key_prefix=prefix))
        db.commit()
        KEY_FILE.write_text(raw_key)
        return brand, raw_key

    brand = Brand(name=DEMO_BRAND_NAME)
    db.add(brand)
    db.commit()
    db.refresh(brand)

    raw_key, prefix, key_hash = generate_api_key()
    db.add(ApiKey(brand_id=brand.id, key_hash=key_hash, key_prefix=prefix))
    db.commit()
    KEY_FILE.write_text(raw_key)
    return brand, raw_key


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        brand, raw_key = get_or_create_demo_brand(db)

        seeded_ids = [s.id for s in db.query(ShopperSession).filter(
            ShopperSession.brand_id == brand.id, ShopperSession.is_seed.is_(True)
        )]
        if seeded_ids:
            db.query(Score).filter(Score.session_id.in_(seeded_ids)).delete(synchronize_session=False)
            db.query(ShopperSession).filter(ShopperSession.id.in_(seeded_ids)).delete(synchronize_session=False)
            db.commit()

        count = 0
        for customer_id, fields in iter_historical_sessions(CSV_PATH):
            session = ShopperSession(
                id=f"seed_{customer_id}",
                brand_id=brand.id,
                is_seed=True,
                **fields,
            )
            db.add(session)
            db.flush()  # so scoring.score_session sees a consistent row if it re-reads

            session_attrs = {c.name: getattr(session, c.name) for c in ShopperSession.__table__.columns}
            score, segment, top_features = scoring.score_session(session_attrs)
            db.add(Score(brand_id=brand.id, session_id=session.id, score=score, segment=segment, top_features=top_features))
            count += 1

        db.commit()
        print(f"Seeded {count} historical sessions for brand '{DEMO_BRAND_NAME}' (brand_id={brand.id})")
        print(f"Demo API key (also saved to {KEY_FILE}): {raw_key}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
