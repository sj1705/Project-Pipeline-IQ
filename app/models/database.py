from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

# Create engine — this is the connection to PostgreSQL
engine = create_engine(settings.database_url)

# SessionLocal — each request gets its own session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base — all our table models will inherit from this
Base = declarative_base()


# Dependency — gives each API request a DB session, then closes it
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()