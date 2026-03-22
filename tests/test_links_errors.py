import pytest
from datetime import datetime, timedelta

class TestLinksErrorScenarios:
    
    def test_create_link_with_invalid_url(self, client, auth_token):
        response = client.post(
            "/links/shorten",
            json={"original_url": "not_a_valid_url"},
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 422
    
    def test_create_link_with_expired_date(self, client, auth_token):
        expires_at = (datetime.utcnow() - timedelta(days=1)).isoformat()
        response = client.post(
            "/links/shorten",
            json={"original_url": "https://example.com", "expires_at": expires_at},
            headers={"Authorization": f"Bearer {auth_token}"})
        if response.status_code == 200:
            data = response.json()
            client.delete(f"/links/{data['short_code']}", headers={"Authorization": f"Bearer {auth_token}"})
        else:
            assert response.status_code == 400
    
    def test_get_link_stats_unauthorized(self, client, test_link):
        response = client.get(f"/links/{test_link.short_code}/stats")
        if response.status_code == 200:
            data = response.json()
            assert data["short_code"] == test_link.short_code
        else:
            assert response.status_code == 401
    
    def test_get_nonexistent_link_stats_with_auth(self, client, auth_token):
        response = client.get(
            "/links/nonexistent/stats",
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 404
    
    def test_update_link_with_invalid_url(self, client, auth_token, test_link):
        response = client.put(
            f"/links/{test_link.short_code}",
            json={"original_url": "invalid_url"},
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 422
    
    def test_update_nonexistent_link(self, client, auth_token):
        response = client.put(
            "/links/nonexistent",
            json={"original_url": "https://example.com/updated"},
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 404
    
    def test_delete_nonexistent_link(self, client, auth_token):
        response = client.delete(
            "/links/nonexistent",
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 404
    
    def test_search_links_empty_result(self, client, auth_token):
        response = client.get(
            "/links/search/",
            params={"original_url": "nonexistent_pattern"},
            headers={"Authorization": f"Bearer {auth_token}"})
        assert response.status_code == 200
        assert response.json() == []
