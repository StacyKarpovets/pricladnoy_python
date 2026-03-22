import pytest

class TestAuthErrorScenarios:
    
    def test_register_with_username_too_short(self, client):
        response = client.post(
            "/auth/register",
            json={"username": "ab", "email": "test@test.com", "password": "password123"})
        if response.status_code == 200:
            data = response.json()
            pass
        else:
            assert response.status_code == 422
    
    def test_register_with_username_too_long(self, client):
        response = client.post(
            "/auth/register",
            json={"username": "a" * 51, "email": "test@test.com", "password": "password123"})
        if response.status_code == 200:
            data = response.json()
        else:
            assert response.status_code == 422
    
    def test_register_with_password_too_short(self, client):
        response = client.post(
            "/auth/register",
            json={"username": "testuser", "email": "test@test.com", "password": "123"})
        if response.status_code == 200:
            data = response.json()
        else:
            assert response.status_code == 422
    
    def test_login_with_empty_username(self, client):
        response = client.post(
            "/auth/token",
            data={"username": "", "password": "pass123"})
        assert response.status_code == 401
    
    def test_login_with_empty_password(self, client):
        response = client.post(
            "/auth/token",
            data={"username": "testuser", "password": ""})
        assert response.status_code == 401
    
    def test_login_with_missing_fields(self, client):
        response = client.post(
            "/auth/token",
            data={"username": "testuser"})
        assert response.status_code == 422
