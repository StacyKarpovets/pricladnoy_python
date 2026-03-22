from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import redis
import os
from urllib.parse import urlparse

if os.environ.get("PYTEST_RUNNING"):
    DATABASE_URL = "sqlite:///./test.db"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_size=5,
        max_overflow=10)
else:
    DATABASE_URL = os.getenv("DATABASE_URL")
    if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    if not DATABASE_URL:
        POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
        POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
        POSTGRES_DB = os.getenv("POSTGRES_DB", "urlshortener")
        POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
        
        DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    engine = create_engine(
        DATABASE_URL,
        echo=True,
        pool_size=5,
        max_overflow=10)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

if os.environ.get("PYTEST_RUNNING"):
    class MockRedis:
        def __init__(self):
            self._data = {}
            self._counters = {}
        
        def get(self, key):
            return self._data.get(key)
        
        def setex(self, key, time, value):
            self._data[key] = value
        
        def incr(self, key):
            self._counters[key] = self._counters.get(key, 0) + 1
            return self._counters[key]
        
        def delete(self, *keys):
            for key in keys:
                self._data.pop(key, None)
                self._counters.pop(key, None)
    
    redis_client = MockRedis()
else:
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_url = urlparse(REDIS_URL)
    REDIS_HOST = redis_url.hostname or "localhost"
    REDIS_PORT = redis_url.port or 6379
    REDIS_PASSWORD = redis_url.password
    REDIS_DB = 0

    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        decode_responses=True,
        ssl=REDIS_URL.startswith("rediss://"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_redis():
    return redis_client

__all__ = ['engine', 'SessionLocal', 'Base', 'get_db', 'get_redis', 'redis_client']

if not os.environ.get("PYTEST_RUNNING"):
    __all__.append('REDIS_URL')

