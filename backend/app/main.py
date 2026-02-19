from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes import video, analysis


app = FastAPI(
    title=settings.APP_NAME,
    description="AI-Driven Golf Swing Analysis API with PyTorch + MuJoCo backend",
    version="2.0.0",  # Phase 2A
    debug=settings.DEBUG
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
app.include_router(video.router)
app.include_router(analysis.router)  # Phase 2A: Golf swing analysis


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "AI-Driven Golf Swing Analysis API",
        "version": "2.0.0",
        "phase": "Phase 2A",
        "status": "running",
        "features": [
            "3D Pose Estimation (MediaPipe)",
            "Physics Simulation (MuJoCo)",
            "Biomechanics Analysis"
        ],
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
