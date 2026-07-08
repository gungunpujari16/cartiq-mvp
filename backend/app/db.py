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

is_sqlite = settings.database_url.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}
# pool_pre_ping + pool_recycle: managed Postgres (Render/Neon free tiers)
# silently drops idle connections after a short timeout. Without these,
# SQLAlchemy hands out stale connections that fail mid-request and the pool
# exhausts under sustained traffic (observed as "QueuePool limit ... reached"
# after ~270 sequential requests when seeding the deployed demo data).
engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=280,
)
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
