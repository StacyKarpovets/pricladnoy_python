import sys
import traceback
import os

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import RedirectResponse
    from app.routers import links, auth
    from app.database import engine, Base
except Exception as e:
    print("="*50, file=sys.stderr)
    print("ERROR DURING IMPORT:", file=sys.stderr)
    print("="*50, file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("="*50, file=sys.stderr)
    raise

try:
    app = FastAPI(
        title="URL Shortener Service",
        description="API для сокращения ссылок с аналитикой и управлением",
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

    print("Adding routers...", file=sys.stderr)
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(links.router, prefix="/links", tags=["Links"])
    print("Routers added successfully", file=sys.stderr)

    @app.on_event("startup")
    def startup():
        print("Creating database tables...", file=sys.stderr)
        try:
            Base.metadata.create_all(bind=engine)
            print("Database tables created/verified", file=sys.stderr)
        except Exception as e:
            print("ERROR CREATING TABLES:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise

    @app.on_event("shutdown")
    def shutdown():
        print("Shutting down...", file=sys.stderr)
        engine.dispose()

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

    print("Application startup complete", file=sys.stderr)

except Exception as e:
    print("="*50, file=sys.stderr)
    print("ERROR DURING APP CREATION:", file=sys.stderr)
    print("="*50, file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("="*50, file=sys.stderr)
    raise
