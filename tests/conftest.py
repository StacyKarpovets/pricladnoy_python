import pytest
import sys
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import redis
from unittest.mock import Mock
import app.main
import importlib
importlib.reload(app.main)
from app.main import app
from app.database import Base, get_db, redis_client
from app.models import User, Link
from app.auth import get_password_hash

TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class TestRedis:
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
    
    def get_counter(self, key):
        return self._counters.get(key, 0)


@pytest.fixture
def test_redis():
    return TestRedis()


@pytest.fixture
def test_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(test_db, test_redis):
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    def override_get_redis():
        return test_redis
    
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[redis_client] = override_get_redis
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(test_db):
    user = User(
        username="tester",
        email="tester@example.com",
        hashed_password=get_password_hash("secret123"),
        is_active=True
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def auth_token(client, test_user):
    response = client.post(
        "/auth/token",
        data={"username": "tester", "password": "secret123"}
    )
    return response.json()["access_token"]


@pytest.fixture
def test_link(test_db, test_user):
    link = Link(
        original_url="https://example.com",
        short_code="test123",
        user_id=test_user.id,
        clicks=0,
        is_active=True
    )
    test_db.add(link)
    test_db.commit()
    test_db.refresh(link)
    return link
