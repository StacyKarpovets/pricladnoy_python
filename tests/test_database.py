import pytest
import os

class TestDatabaseFunctions:
    
    def test_get_db_returns_session(self, test_db):
        from app.database import get_db
        db_gen = get_db()
        db = next(db_gen)
        assert db is not None
        try:
            next(db_gen)
        except StopIteration:
            pass
    
    def test_get_redis_returns_client(self):
        from app.database import get_redis
        redis_client = get_redis()
        assert redis_client is not None
    
    def test_mock_redis_set_and_get(self, test_redis):
        test_redis.setex("test_key", 60, "test_value")
        assert test_redis.get("test_key") == "test_value"
    
    def test_mock_redis_increment(self, test_redis):
        test_redis.incr("counter")
        assert test_redis.get_counter("counter") == 1
        test_redis.incr("counter")
        assert test_redis.get_counter("counter") == 2
    
    def test_mock_redis_delete(self, test_redis):
        test_redis.setex("key_to_delete", 60, "value")
        assert test_redis.get("key_to_delete") == "value"
        test_redis.delete("key_to_delete")
        assert test_redis.get("key_to_delete") is None

class TestDatabaseConfig:
    
    def test_database_url_from_env(self, monkeypatch):
        monkeypatch.delenv("PYTEST_RUNNING", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/testdb")
        
        import importlib
        import app.database
        importlib.reload(app.database)
        
        assert app.database.DATABASE_URL is not None
    
    def test_redis_client_returns_mock_in_test(self):
        from app.database import redis_client
        assert redis_client is not None
        assert hasattr(redis_client, 'get')
        assert hasattr(redis_client, 'setex')
        assert hasattr(redis_client, 'incr')
        assert hasattr(redis_client, 'delete')
    
    def test_mock_redis_operations(self, test_redis):
        test_redis.setex("key", 60, "value")
        assert test_redis.get("key") == "value"
        
        test_redis.incr("counter")
        assert test_redis.get_counter("counter") == 1
        
        test_redis.delete("key")
        assert test_redis.get("key") is None
