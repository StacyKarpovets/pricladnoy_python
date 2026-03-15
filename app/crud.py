from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.sql import func
from datetime import datetime, timedelta, timezone
import random
import string
import json
from typing import Optional, List, Dict, Any

from app.models import Link
from app.schemas import LinkCreate, LinkUpdate
from app.database import get_redis

class LinkCRUD:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.redis = None

    async def _get_redis(self):
        if not self.redis:
            self.redis = await get_redis()
        return self.redis

    @staticmethod
    def generate_short_code(length: int = 6) -> str:
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    async def get_unique_short_code(self) -> str:
        while True:
            code = self.generate_short_code()
            existing = await self.get_link_by_short_code(code)
            if not existing:
                return code

    async def create_link(
        self,
        link_data: LinkCreate,
        user_id: Optional[str] = None
    ) -> Link:
        if link_data.custom_alias:
            existing = await self.get_link_by_short_code(link_data.custom_alias)
            if existing:
                raise ValueError("Custom alias already exists")
            short_code = link_data.custom_alias
        else:
            short_code = await self.get_unique_short_code()

        db_link = Link(
            original_url=str(link_data.original_url),
            short_code=short_code,
            custom_alias=link_data.custom_alias,
            user_id=user_id,
            expires_at=link_data.expires_at
        )

        self.db.add(db_link)
        await self.db.commit()
        await self.db.refresh(db_link)

        await self.cache_link(db_link)

        return db_link

    async def get_link_by_short_code(self, short_code: str) -> Optional[Link]:
        redis_client = await self._get_redis()
        cached = await redis_client.get(f"link:{short_code}")

        if cached:
            try:
                data = json.loads(cached)
                return data
            except:
                pass

        result = await self.db.execute(
            select(Link).where(
                and_(
                    Link.short_code == short_code,
                    Link.is_active == True
                )
            )
        )
        link = result.scalar_one_or_none()

        return link

    async def increment_clicks(self, short_code: str):
        redis_client = await self._get_redis()
        await redis_client.incr(f"clicks:{short_code}")

        result = await self.db.execute(
            select(Link).where(Link.short_code == short_code)
        )
        link = result.scalar_one_or_none()
        if link:
            link.clicks += 1
            link.last_used_at = func.now()
            await self.db.commit()

    async def update_link(
        self,
        short_code: str,
        link_update: LinkUpdate,
        user_id: str
    ) -> Optional[Link]:
        result = await self.db.execute(
            select(Link).where(
                and_(
                    Link.short_code == short_code,
                    Link.is_active == True
                )
            )
        )
        link = result.scalar_one_or_none()

        if not link:
            return None

        if link.user_id and link.user_id != user_id:
            raise PermissionError("You don't have permission to update this link")

        link.original_url = str(link_update.original_url)
        await self.db.commit()
        await self.db.refresh(link)

        await self.cache_link(link)

        return link

    async def delete_link(self, short_code: str, user_id: Optional[str] = None) -> bool:
        result = await self.db.execute(
            select(Link).where(
                and_(
                    Link.short_code == short_code,
                    Link.is_active == True
                )
            )
        )
        link = result.scalar_one_or_none()

        if not link:
            return False

        if user_id and link.user_id and link.user_id != user_id:
            raise PermissionError("You don't have permission to delete this link")

        link.is_active = False
        await self.db.commit()

        redis_client = await self._get_redis()
        await redis_client.delete(f"link:{link.short_code}")
        await redis_client.delete(f"clicks:{link.short_code}")

        return True

    async def get_link_stats(self, short_code: str) -> Optional[Dict[str, Any]]:
        result = await self.db.execute(
            select(Link).where(Link.short_code == short_code)
        )
        link = result.scalar_one_or_none()

        if not link:
            return None

        redis_client = await self._get_redis()
        redis_clicks = await redis_client.get(f"clicks:{short_code}")

        total_clicks = link.clicks
        if redis_clicks:
            total_clicks += int(redis_clicks)

        now = datetime.now(timezone.utc)

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

    async def search_by_original_url(
        self,
        original_url: str,
        user_id: Optional[str] = None
    ) -> List[Link]:
        query = select(Link).where(
            and_(
                Link.original_url.contains(original_url),
                Link.is_active == True
            )
        )

        if user_id:
            query = query.where(Link.user_id == user_id)

        result = await self.db.execute(query)
        return result.scalars().all()

    async def cleanup_expired_links(self):
        result = await self.db.execute(
            select(Link).where(
                and_(
                    Link.expires_at < datetime.now(timezone.utc),
                    Link.is_active == True
                )
            )
        )
        expired_links = result.scalars().all()

        redis_client = await self._get_redis()
        for link in expired_links:
            link.is_active = False
            await redis_client.delete(f"link:{link.short_code}")
            await redis_client.delete(f"clicks:{link.short_code}")

        await self.db.commit()
        return len(expired_links)

    async def cache_link(self, link: Link):
        redis_client = await self._get_redis()

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

        await redis_client.setex(
            f"link:{link.short_code}",
            3600,
            json.dumps(link_data, default=str)
        )

