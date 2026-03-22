import pytest
from datetime import datetime, timedelta

class TestAuthEndpoints:
    
    def test_register_new_user(self, client):
        response = client.post(
            "/auth/register",
            json={"username": "newuser", "email": "newuser@test.com", "password": "pass123"})
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newuser"
        assert "id" in data
    
    def test_register_duplicate_username(self, client, test_user):
        response = client.post(
            "/auth/register",
            json={"username": "tester", "email": "another@test.com", "password": "secret123"})
        assert response.status_code == 400
        assert "already registered" in response.text
    
    def test_login_success(self, client, test_user):
        response = client.post(
            "/auth/token",
            data={"username": "tester", "password": "secret123"})
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_wrong_password(self, client, test_user):
        response = client.post(
            "/auth/token",
            data={"username": "tester", "password": "wrong"})
        assert response.status_code == 401
    
    def test_get_current_user(self, client, auth_token):
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 200
        assert response.json()["username"] == "tester"

class TestLinksEndpoints:
    
    def test_create_link_requires_auth(self, client):
        response = client.post(
            "/links/shorten",
            json={"original_url": "https://example.com"})
        assert response.status_code == 401
    
    def test_create_link_success(self, client, auth_token):
        response = client.post(
            "/links/shorten",
            json={"original_url": "https://example.com"},
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 200
        data = response.json()
        assert data["original_url"] == "https://example.com/"
        assert "short_code" in data
    
    def test_create_link_with_custom_alias(self, client, auth_token):
        response = client.post(
            "/links/shorten",
            json={"original_url": "https://example.com", "custom_alias": "myalias"},
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 200
        assert response.json()["short_code"] == "myalias"
    
    def test_create_link_with_expiration(self, client, auth_token):
        expires_at = (datetime.utcnow() + timedelta(days=7)).isoformat()
        response = client.post(
            "/links/shorten",
            json={"original_url": "https://example.com", "expires_at": expires_at},
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 200
        assert response.json()["expires_at"] is not None
    
    def test_get_link_stats(self, client, test_link):
        response = client.get(f"/links/{test_link.short_code}/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["short_code"] == test_link.short_code
        assert data["clicks"] == 0
    
    def test_get_nonexistent_link_stats(self, client):
        response = client.get("/links/nonexistent/stats")
        assert response.status_code == 404
    
    def test_redirect_to_url(self, client, test_link):
        response = client.get(f"/{test_link.short_code}", follow_redirects=False)
        assert response.status_code == 307
    
    def test_redirect_nonexistent_url(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_update_link(self, client, auth_token, test_link):
        response = client.put(
            f"/links/{test_link.short_code}",
            json={"original_url": "https://example.com/updated"},
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 200
        assert response.json()["original_url"] == "https://example.com/updated"
    
    def test_update_link_unauthorized(self, client, test_link):
        response = client.put(
            f"/links/{test_link.short_code}",
            json={"original_url": "https://example.com/updated"})
        assert response.status_code == 401
    
    def test_delete_link(self, client, auth_token, test_link):
        response = client.delete(
            f"/links/{test_link.short_code}",
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 200
        assert response.json()["message"] == "Link deleted successfully"
    
    def test_search_links(self, client, auth_token, test_link):
        response = client.get(
            "/links/search/",
            params={"original_url": "example"},
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 200
        assert isinstance(response.json(), list)

class TestGeneralEndpoints:
    
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "health_check" in data
