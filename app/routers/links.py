from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime

from app.database import get_db
from app import crud, auth, schemas
from app.auth import get_current_active_user
from app.models import User, Link

router = APIRouter()

@router.post("/shorten", response_model=schemas.LinkResponse)
def create_short_link(
    request: Request,
    link_data: schemas.LinkCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(auth.get_current_active_user)
):
    link_crud = crud.LinkCRUD(db)
    
    try:
        user_id = current_user.id if current_user else None
        link = link_crud.create_link(link_data, user_id)
        
        base_url = str(request.base_url).rstrip('/')
        short_url = f"{base_url}/{link.short_code}"
        
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
def redirect_to_url(
    short_code: str,
    db: Session = Depends(get_db)
):
    link_crud = crud.LinkCRUD(db)
    link = link_crud.get_link_by_short_code(short_code)
    
    if not link or isinstance(link, dict):
        link = db.query(Link).filter(Link.short_code == short_code).first()
    
    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Link not found or expired"
        )
    
    if link.expires_at and link.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Link has expired"
        )
    
    link_crud.increment_clicks(short_code)
    
    return RedirectResponse(url=link.original_url)

@router.get("/{short_code}/stats", response_model=schemas.LinkStats)
def get_link_stats(
    short_code: str,
    db: Session = Depends(get_db)
):
    link_crud = crud.LinkCRUD(db)
    stats = link_crud.get_link_stats(short_code)
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Link not found"
        )
    
    link = db.query(Link).filter(Link.short_code == short_code).first()
    
    if link:
        base_url = "http://localhost:8000"
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
def update_link(
    short_code: str,
    link_update: schemas.LinkUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    link_crud = crud.LinkCRUD(db)
    
    try:
        link = link_crud.update_link(short_code, link_update, current_user.id)
        if not link:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Link not found"
            )
        
        base_url = "http://localhost:8000"
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
def delete_link(
    short_code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    link_crud = crud.LinkCRUD(db)
    
    try:
        deleted = link_crud.delete_link(short_code, current_user.id)
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
def search_links(
    original_url: str = Query(..., description="Original URL to search for"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(auth.get_current_active_user)
):
    link_crud = crud.LinkCRUD(db)
    user_id = current_user.id if current_user else None
    links = link_crud.search_by_original_url(original_url, user_id)
    
    base_url = "http://localhost:8000"
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
def cleanup_expired_links(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    link_crud = crud.LinkCRUD(db)
    count = link_crud.cleanup_expired_links()
    return {"message": f"Cleaned up {count} expired links"}
