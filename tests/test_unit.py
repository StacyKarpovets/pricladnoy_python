import pytest
from datetime import datetime, timedelta
from app.crud import LinkCRUD
from app.schemas import LinkCreate
from app.models import Link


class TestShortCodeGeneration:
    
    def test_generate_short_code_default_length(self):
        code = LinkCRUD.generate_short_code()
        assert len(code) == 6
        assert code.isalnum()
    
    def test_generate_short_code_custom_length(self):
        code = LinkCRUD.generate_short_code(8)
        assert len(code) == 8
    
    def test_generate_short_code_randomness(self):
        codes = [LinkCRUD.generate_short_code() for _ in range(10)]
        assert len(set(codes)) == 10


class TestLinkCreation:
    
    def test_create_simple_link(self, test_db, test_user):
        crud = LinkCRUD(test_db)
        link_data = LinkCreate(original_url="https://example.com")
        
        link = crud.create_link(link_data, test_user.id)
        
        assert link.original_url == "https://example.com/"
        assert link.short_code is not None
        assert len(link.short_code) == 6
        assert link.user_id == test_user.id
        assert link.clicks == 0
        assert link.is_active is True
    
    def test_create_link_with_custom_alias(self, test_db, test_user):
        crud = LinkCRUD(test_db)
        link_data = LinkCreate(
            original_url="https://example.com",
            custom_alias="myalias_unique")
        
        link = crud.create_link(link_data, test_user.id)
        
        assert link.short_code == "myalias_unique"
        assert link.custom_alias == "myalias_unique"
    
    def test_create_link_duplicate_alias_fails(self, test_db, test_user):
        crud = LinkCRUD(test_db)
        link_data = LinkCreate(
            original_url="https://example.com",
            custom_alias="duplicate")
        crud.create_link(link_data, test_user.id)
        
        with pytest.raises(ValueError, match="Custom alias already exists"):
            crud.create_link(link_data, test_user.id)
    
    def test_create_link_with_expiration(self, test_db, test_user):
        crud = LinkCRUD(test_db)
        expires_at = datetime.utcnow() + timedelta(days=7)
        link_data = LinkCreate(
            original_url="https://example.com",
            expires_at=expires_at)
        
        link = crud.create_link(link_data, test_user.id)
        
        assert link.expires_at is not None


class TestLinkRetrieval:
    
    def test_get_link_by_short_code(self, test_db, test_link):
        crud = LinkCRUD(test_db)
        
        link = crud.get_link_by_short_code(test_link.short_code)
        
        assert link is not None
        assert link.short_code == test_link.short_code
    
    def test_get_nonexistent_link_returns_none(self, test_db):
        crud = LinkCRUD(test_db)
        
        link = crud.get_link_by_short_code("nonexistent")
        
        assert link is None


class TestClickCounting:
    
    def test_increment_clicks(self, test_db, test_link):
        crud = LinkCRUD(test_db)
        
        crud.increment_clicks(test_link.short_code)
        link = crud.get_link_by_short_code(test_link.short_code)
        assert link.clicks == 1
        
        crud.increment_clicks(test_link.short_code)
        link = crud.get_link_by_short_code(test_link.short_code)
        assert link.clicks == 2


class TestLinkUpdate:
    
    def test_update_link_url(self, test_db, test_user, test_link):
        crud = LinkCRUD(test_db)
        new_url = LinkCreate(original_url="https://example.com/new")
        
        updated = crud.update_link(test_link.short_code, new_url, test_user.id)
        
        assert updated.original_url == "https://example.com/new"
    
    def test_update_link_wrong_user_fails(self, test_db, test_link):
        crud = LinkCRUD(test_db)
        new_url = LinkCreate(original_url="https://example.com/new")
        
        with pytest.raises(PermissionError, match="permission to update"):
            crud.update_link(test_link.short_code, new_url, "wrong-user-id")


class TestLinkDeletion:
    
    def test_delete_link(self, test_db, test_user, test_link):
        crud = LinkCRUD(test_db)
        
        result = crud.delete_link(test_link.short_code, test_user.id)
        
        assert result is True
        link = crud.get_link_by_short_code(test_link.short_code)
        assert link is None
    
    def test_delete_wrong_user_fails(self, test_db, test_link):
        crud = LinkCRUD(test_db)
        
        with pytest.raises(PermissionError, match="permission to delete"):
            crud.delete_link(test_link.short_code, "wrong-user-id")


class TestLinkStats:
    
    def test_get_link_stats(self, test_db, test_link):
        crud = LinkCRUD(test_db)
        
        stats = crud.get_link_stats(test_link.short_code)
        
        assert stats is not None
        assert stats["original_url"] == test_link.original_url
        assert stats["short_code"] == test_link.short_code
        assert "clicks" in stats
        assert "created_at" in stats
        assert "is_expired" in stats
        assert isinstance(stats["is_expired"], bool)
    
    def test_expired_link_marked_correctly(self, test_db):
        crud = LinkCRUD(test_db)
        expired_link = Link(
            original_url="https://example.com/expired",
            short_code="expired",
            expires_at=datetime.utcnow() - timedelta(days=1),
            is_active=True)
        test_db.add(expired_link)
        test_db.commit()
        
        stats = crud.get_link_stats("expired")
        
        assert stats["is_expired"] is True


class TestSearchAndCleanup:
    
    def test_search_by_original_url(self, test_db, test_user, test_link):
        crud = LinkCRUD(test_db)
        
        results = crud.search_by_original_url("example", test_user.id)
        
        assert len(results) >= 1
        assert any("example" in link.original_url for link in results)
    
    def test_cleanup_expired_links(self, test_db):
        crud = LinkCRUD(test_db)
        
        expired_link = Link(
            original_url="https://example.com/expired",
            short_code="expired",
            expires_at=datetime.utcnow() - timedelta(days=1),
            is_active=True)
        test_db.add(expired_link)
        test_db.commit()
        
        count = crud.cleanup_expired_links()
        
        assert count >= 1
        expired = crud.get_link_by_short_code("expired")
        assert expired is None
