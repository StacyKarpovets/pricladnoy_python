from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime, timedelta
import random
import string
import json
from typing import Optional, List, Dict, Any

from app.models import Link
from app.schemas import LinkCreate, LinkUpdate
from app.database import redis_client

class LinkCRUD:
    def __init__(self, db: Session):
        self.db = db
        self.redis = redis_client

    @staticmethod
    def generate_short_code(length: int = 6) -> str:
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    def get_unique_short_code(self) -> str:
        while True:
            code = self.generate_short_code()
            existing = self.get_link_by_short_code(code)
            if not existing:
                return code

    def create_link(
        self,
        link_data: LinkCreate,
        user_id: Optional[str] = None
    ) -> Link:
        if link_data.custom_alias:
            existing = self.get_link_by_short_code(link_data.custom_alias)
            if existing:
                raise ValueError("Custom alias already exists")
            short_code = link_data.custom_alias
        else:
            short_code = self.get_unique_short_code()

        db_link = Link(
            original_url=str(link_data.original_url),
            short_code=short_code,
            custom_alias=link_data.custom_alias,
            user_id=user_id,
            expires_at=link_data.expires_at
        )

        self.db.add(db_link)
        self.db.commit()
        self.db.refresh(db_link)

        self.cache_link(db_link)

        return db_link

    def get_link_by_short_code(self, short_code: str) -> Optional[Link]:
        cached = self.redis.get(f"link:{short_code}")
        if cached:
            try:
                data = json.loads(cached)
                return data
            except:
                pass

        return self.db.query(Link).filter(
            and_(
                Link.short_code == short_code,
                Link.is_active == True
            )
        ).first()

    def increment_clicks(self, short_code: str):
        self.redis.incr(f"clicks:{short_code}")

        link = self.db.query(Link).filter(Link.short_code == short_code).first()
        if link:
            link.clicks += 1
            link.last_used_at = datetime.utcnow()
            self.db.commit()

    def update_link(
        self,
        short_code: str,
        link_update: LinkUpdate,
        user_id: str
    ) -> Optional[Link]:
        link = self.db.query(Link).filter(
            and_(
                Link.short_code == short_code,
                Link.is_active == True
            )
        ).first()

        if not link:
            return None

        if link.user_id and link.user_id != user_id:
            raise PermissionError("You don't have permission to update this link")

        link.original_url = str(link_update.original_url)
        self.db.commit()
        self.db.refresh(link)

        self.cache_link(link)

        return link

    def delete_link(self, short_code: str, user_id: Optional[str] = None) -> bool:
        link = self.db.query(Link).filter(
            and_(
                Link.short_code == short_code,
                Link.is_active == True
            )
        ).first()

        if not link:
            return False

        if user_id and link.user_id and link.user_id != user_id:
            raise PermissionError("You don't have permission to delete this link")

        link.is_active = False
        self.db.commit()

        self.redis.delete(f"link:{short_code}")
        self.redis.delete(f"clicks:{short_code}")

        return True

    def get_link_stats(self, short_code: str) -> Optional[Dict[str, Any]]:
        link = self.db.query(Link).filter(Link.short_code == short_code).first()

        if not link:
            return None

        redis_clicks = self.redis.get(f"clicks:{short_code}")

        total_clicks = link.clicks
        if redis_clicks:
            total_clicks += int(redis_clicks)

        now = datetime.utcnow()

        is_expired = False
        if link.expires_at:
            is_expired = link.expires_at < now

        stats = {
            "original_url": link.original_url,
            "short_code": link.short_code,
            "clicks": total_clicks,
            "created_at": link.created_at.isoformat() if link.created_at else None,
            "last_used_at": link.last_used_at.isoformat() if link.last_used_at else None,
            "expires_at": link.expires_at.isoformat() if link.expires_at else None,
            "is_expired": is_expired,
            "days_since_creation": (now - link.created_at).days if link.created_at else None,
            "is_active": link.is_active
        }

        return stats

    def search_by_original_url(
        self,
        original_url: str,
        user_id: Optional[str] = None
    ) -> List[Link]:
        query = self.db.query(Link).filter(
            and_(
                Link.original_url.contains(original_url),
                Link.is_active == True
            )
        )

        if user_id:
            query = query.filter(Link.user_id == user_id)

        return query.all()

    def cleanup_expired_links(self):
        expired_links = self.db.query(Link).filter(
            and_(
                Link.expires_at < datetime.utcnow(),
                Link.is_active == True
            )
        ).all()

        for link in expired_links:
            link.is_active = False
            self.redis.delete(f"link:{link.short_code}")
            self.redis.delete(f"clicks:{link.short_code}")

        self.db.commit()
        return len(expired_links)

    def cache_link(self, link: Link):
        link_data = {
            "id": link.id,
            "original_url": link.original_url,
            "short_code": link.short_code,
            "clicks": link.clicks,
            "created_at": link.created_at.isoformat() if link.created_at else None,
            "last_used_at": link.last_used_at.isoformat() if link.last_used_at else None,
            "expires_at": link.expires_at.isoformat() if link.expires_at else None,
            "is_active": link.is_active,
            "user_id": link.user_id
        }

        self.redis.setex(
            f"link:{link.short_code}",
            3600,
            json.dumps(link_data, default=str)
        )
