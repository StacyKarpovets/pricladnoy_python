from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import os

from app.database import engine, Base
from app.routers import links, auth

app = FastAPI(
    title="URL Shortener Service",
    description="API для сокращения ссылок",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(links.router, prefix="/links", tags=["Links"])

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    print("Database tables created/verified")

@app.on_event("shutdown")
def shutdown():
    engine.dispose()
    print("Shutting down...")

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
