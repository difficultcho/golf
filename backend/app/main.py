from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routes import video, analysis, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create database tables
    await init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-Driven Golf Swing Analysis API with PyTorch + MuJoCo backend. Deployed via CI/CD pipeline.",
    version="2.2.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# CORS middleware for WeChat mini program
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(video.router)
app.include_router(analysis.router)


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "AI-Driven Golf Swing Analysis API",
        "version": "2.2.0",
        "phase": "Phase 2B - User System",
        "status": "running",
        "features": [
            "WeChat Login & JWT Auth",
            "Role-based Access Control",
            "3D Pose Estimation (MediaPipe)",
            "Physics Simulation (MuJoCo)",
            "Biomechanics Analysis",
        ],
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
