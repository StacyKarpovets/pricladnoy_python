import pytest

class TestMainErrorHandling:
    
    def test_root_endpoint_returns_correct_structure(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
        assert "redoc" in data
        assert "health_check" in data
    
    def test_root_endpoint_with_different_base_urls(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["docs"], str)
        assert isinstance(data["redoc"], str)
        assert isinstance(data["health_check"], str)
    
    def test_health_check_returns_correct_data(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "environment" in data
    
    def test_health_check_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_redirect_with_post_method(self, client, test_link):
        response = client.post(f"/{test_link.short_code}")
        assert response.status_code == 405
    
    def test_redirect_with_put_method(self, client, test_link):
        response = client.put(f"/{test_link.short_code}")
        assert response.status_code == 405
    
    def test_redirect_with_delete_method(self, client, test_link):
        response = client.delete(f"/{test_link.short_code}")
        assert response.status_code == 405
    
    def test_redirect_with_patch_method(self, client, test_link):
        response = client.patch(f"/{test_link.short_code}")
        assert response.status_code == 405
    
    def test_redirect_nonexistent_link(self, client):
        response = client.get("/nonexistent123")
        assert response.status_code == 404
    
    def test_redirect_with_slash(self, client):
        response = client.get("/")
        assert response.status_code == 200
    
    def test_redirect_with_methods_multiple(self, client, test_link):
        methods = ["POST", "PUT", "DELETE", "PATCH"]
        for method in methods:
            if method == "POST":
                response = client.post(f"/{test_link.short_code}")
            elif method == "PUT":
                response = client.put(f"/{test_link.short_code}")
            elif method == "DELETE":
                response = client.delete(f"/{test_link.short_code}")
            else:
                response = client.patch(f"/{test_link.short_code}")
            assert response.status_code == 405
