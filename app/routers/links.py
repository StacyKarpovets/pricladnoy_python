from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from datetime import datetime

from app.database import get_db
from app import crud, auth, schemas
from app.auth import get_current_active_user
from app.models import User

router = APIRouter()

@router.post("/shorten", response_model=schemas.LinkResponse)
async def create_short_link(
    request: Request,
    link_data: schemas.LinkCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(auth.get_current_active_user)
):
    """
    Create a short link
    
    - **original_url**: The URL to shorten
    - **custom_alias**: Optional custom alias (must be unique)
    - **expires_at**: Optional expiration date (ISO format)
    """
    link_crud = crud.LinkCRUD(db)
    
    try:
        user_id = current_user.id if current_user else None
        link = await link_crud.create_link(link_data, user_id)
        
        # Generate full short URL
        base_url = str(request.base_url).rstrip('/')
        short_url = f"{base_url}/{link.short_code}"
        
        # Create response
        response = schemas.LinkResponse(
            id=link.id,
            original_url=link.original_url,
            short_code=link.short_code,
            custom_alias=link.custom_alias,
            clicks=link.clicks,
            created_at=link.created_at,
            last_used_at=link.last_used_at,
            expires_at=link.expires_at,
            is_active=link.is_active,
            user_id=link.user_id,
            short_url=short_url
        )
        
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/{short_code}")
async def redirect_to_url(
    short_code: str,
    db: AsyncSession = Depends(get_db)
):
    """Redirect to original URL"""
    link_crud = crud.LinkCRUD(db)
    link = await link_crud.get_link_by_short_code(short_code)
    
    if not link or isinstance(link, dict):
        # Try to get from database if not in cache
        from app.models import Link
        result = await db.execute(
            select(Link).where(Link.short_code == short_code)
        )
        link = result.scalar_one_or_none()
    
    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Link not found or expired"
        )
    
    # Check if expired
    if link.expires_at and link.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Link has expired"
        )
    
    # Increment clicks asynchronously
    await link_crud.increment_clicks(short_code)
    
    return RedirectResponse(url=link.original_url)

@router.get("/{short_code}/stats", response_model=schemas.LinkStats)
async def get_link_stats(
    short_code: str,
    db: AsyncSession = Depends(get_db)
):
    """Get statistics for a short link"""
    link_crud = crud.LinkCRUD(db)
    stats = await link_crud.get_link_stats(short_code)
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Link not found"
        )
    
    # Get link to create full response
    from app.models import Link
    result = await db.execute(
        select(Link).where(Link.short_code == short_code)
    )
    link = result.scalar_one_or_none()
    
    if link:
        base_url = "https://your-app.onrender.com"  # Замените на ваш URL
        short_url = f"{base_url}/{link.short_code}"
        
        return schemas.LinkStats(
            id=link.id,
            original_url=stats["original_url"],
            short_code=stats["short_code"],
            custom_alias=link.custom_alias,
            clicks=stats["clicks"],
            created_at=link.created_at,
            last_used_at=link.last_used_at,
            expires_at=link.expires_at,
            is_active=link.is_active,
            user_id=link.user_id,
            short_url=short_url,
            days_since_creation=stats["days_since_creation"],
            is_expired=stats["is_expired"]
        )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Link not found"
    )

@router.put("/{short_code}", response_model=schemas.LinkResponse)
async def update_link(
    short_code: str,
    link_update: schemas.LinkUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a short link (authenticated users only)"""
    link_crud = crud.LinkCRUD(db)
    
    try:
        link = await link_crud.update_link(short_code, link_update, current_user.id)
        if not link:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Link not found"
            )
        
        base_url = "https://your-app.onrender.com"  # Замените на ваш URL
        short_url = f"{base_url}/{link.short_code}"
        
        return schemas.LinkResponse(
            id=link.id,
            original_url=link.original_url,
            short_code=link.short_code,
            custom_alias=link.custom_alias,
            clicks=link.clicks,
            created_at=link.created_at,
            last_used_at=link.last_used_at,
            expires_at=link.expires_at,
            is_active=link.is_active,
            user_id=link.user_id,
            short_url=short_url
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )

@router.delete("/{short_code}")
async def delete_link(
    short_code: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a short link (authenticated users only)"""
    link_crud = crud.LinkCRUD(db)
    
    try:
        deleted = await link_crud.delete_link(short_code, current_user.id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Link not found"
            )
        return {"message": "Link deleted successfully"}
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )

@router.get("/search/", response_model=List[schemas.LinkResponse])
async def search_links(
    original_url: str = Query(..., description="Original URL to search for"),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(auth.get_current_active_user)
):
    """Search links by original URL"""
    link_crud = crud.LinkCRUD(db)
    user_id = current_user.id if current_user else None
    links = await link_crud.search_by_original_url(original_url, user_id)
    
    base_url = "https://your-app.onrender.com"  # Замените на ваш URL
    result = []
    for link in links:
        short_url = f"{base_url}/{link.short_code}"
        result.append(schemas.LinkResponse(
            id=link.id,
            original_url=link.original_url,
            short_code=link.short_code,
            custom_alias=link.custom_alias,
            clicks=link.clicks,
            created_at=link.created_at,
            last_used_at=link.last_used_at,
            expires_at=link.expires_at,
            is_active=link.is_active,
            user_id=link.user_id,
            short_url=short_url
        ))
    
    return result

@router.post("/admin/cleanup/expired")
async def cleanup_expired_links(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Admin: Clean up expired links"""
    link_crud = crud.LinkCRUD(db)
    count = await link_crud.cleanup_expired_links()
    return {"message": f"Cleaned up {count} expired links"}
