from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import os

from app.database import engine, Base
from app.routers import links, auth

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created/verified")
    yield
    # Shutdown
    await engine.dispose()
    print("Shutting down...")

app = FastAPI(
    title="URL Shortener Service",
    description="API для сокращения ссылок с аналитикой и управлением",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(links.router, prefix="/links", tags=["Links"])

# Корневой эндпоинт для редиректа коротких ссылок
@app.get("/{short_code}")
async def redirect_short_link(short_code: str, request: Request):
    return RedirectResponse(url=f"/links/{short_code}")

@app.get("/")
async def root():
    base_url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
    return {
        "message": "Welcome to URL Shortener Service",
        "docs": f"{base_url}/docs",
        "redoc": f"{base_url}/redoc",
        "health_check": f"{base_url}/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": os.getenv("RENDER_ENV", "development")
    }
