from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime
from typing import Optional, List

# Link schemas
class LinkBase(BaseModel):
    original_url: HttpUrl

class LinkCreate(LinkBase):
    custom_alias: Optional[str] = Field(None, min_length=3, max_length=20, pattern="^[a-zA-Z0-9_-]+$")
    expires_at: Optional[datetime] = None

class LinkUpdate(BaseModel):
    original_url: HttpUrl

class LinkResponse(BaseModel):
    id: str
    original_url: str
    short_code: str
    custom_alias: Optional[str]
    clicks: int
    created_at: datetime
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    is_active: bool
    user_id: Optional[str]
    short_url: str
    
    class Config:
        from_attributes = True

class LinkStats(LinkResponse):
    days_since_creation: Optional[int]
    is_expired: bool

class LinkSearch(BaseModel):
    results: List[LinkResponse]
    total: int

# User schemas
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None
