"""
Database connection and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
from config.settings import get_settings

# Base class for declarative models
Base = declarative_base()

# Global engine and session factory
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        
        # Special handling for SQLite
        if settings.database_url.startswith("sqlite"):
            _engine = create_engine(
                settings.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=settings.log_level == "DEBUG"
            )
        else:
            _engine = create_engine(
                settings.database_url,
                echo=settings.log_level == "DEBUG"
            )
    
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
    return _SessionLocal


def get_db():
    """
    Get a database session.
    Use as a context manager or dependency injection.
    
    Example:
        with get_db() as db:
            # Use db session
            pass
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize the database (create all tables)."""
    from database.models import Chat, Message  # Import models
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def reset_db():
    """Reset the database (drop and recreate all tables). USE WITH CAUTION!"""
    from database.models import Chat, Message  # Import models
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
