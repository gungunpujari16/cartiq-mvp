import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Brand, ShopperSession


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture()
def brand(db_session):
    b = Brand(name="Test Brand", discount_score_threshold=40.0, discount_min_cart_value=50.0, discount_max_pct=15.0)
    db_session.add(b)
    db_session.commit()
    db_session.refresh(b)
    return b


@pytest.fixture()
def shopper_session(db_session, brand):
    s = ShopperSession(
        id="sess_1",
        brand_id=brand.id,
        cart_value=100.0,
        return_customer=False,
        discount_used=False,
    )
    db_session.add(s)
    db_session.commit()
    db_session.refresh(s)
    return s
