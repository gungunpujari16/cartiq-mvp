"""
DB engine/session setup.

TRD mapping: TRD specifies Cassandra (event store) + Snowflake (analytics
warehouse), each physically partitioned per tenant. At MVP scale a single
Postgres (or SQLite locally) with a brand_id column on every table gives the
same *logical* isolation guarantee (every query is filtered by the caller's
brand_id, resolved from their API key in auth.py) without the operational
cost of separate infra per tenant.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app import models  # noqa: F401 ensure models are registered

    Base.metadata.create_all(bind=engine)
