from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./cloudsense.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def ensure_schema():
    """Small SQLite compatibility migration for new CloudSense columns."""
    from sqlalchemy import inspect, text
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    if "aws_resources" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("aws_resources")}
    with engine.begin() as conn:
        if "monthly_cost" not in existing:
            conn.execute(text("ALTER TABLE aws_resources ADD COLUMN monthly_cost FLOAT"))
        if "cost_currency" not in existing:
            conn.execute(text("ALTER TABLE aws_resources ADD COLUMN cost_currency VARCHAR(10)"))
