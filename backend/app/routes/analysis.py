"""
Analysis API Routes - Phase 2A

Endpoints for golf swing analysis:
- POST /{video_id} - Run analysis on uploaded video
- GET /result/{analysis_id} - Get analysis results
- GET /visualization/{analysis_id} - Download annotated video
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models.db_models import User, Video as VideoDB, Analysis as AnalysisDB, UserRole
from app.models.video import AnalysisResult
from app.services.golf_analysis_service import GolfAnalysisService


router = APIRouter(prefix="/api/analysis", tags=["analysis"])
service = GolfAnalysisService()


def _check_owner(user_id: int, current_user: User):
    """Raise 404 if current user doesn't own the resource (admins bypass)."""
    if current_user.role != UserRole.admin and user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Resource not found")


@router.post("/{video_id}", response_model=AnalysisResult)
async def analyze_video(
    video_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Run Phase 2A analysis on uploaded video."""
    # Verify video ownership
    result = await db.execute(select(VideoDB).where(VideoDB.id == video_id))
    video = result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    _check_owner(video.user_id, current_user)

    try:
        analysis_result = await service.analyze_video(video_id)

        # Save analysis record to DB
        analysis_record = AnalysisDB(
            id=analysis_result.analysis_id,
            video_id=video_id,
            user_id=current_user.id,
            status="completed",
            club_head_speed_mph=analysis_result.physics_metrics.club_head_speed_mph,
            x_factor=analysis_result.physics_metrics.x_factor,
            balance_score=analysis_result.physics_metrics.balance_score,
            energy_efficiency=analysis_result.physics_metrics.energy_efficiency,
            swing_duration_sec=analysis_result.physics_metrics.swing_duration_sec,
            processing_time=analysis_result.processing_time,
        )
        db.add(analysis_record)
        await db.commit()

        return analysis_result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Video not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/result/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_result(
    analysis_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Retrieve previously computed analysis result."""
    result = await db.execute(select(AnalysisDB).where(AnalysisDB.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    _check_owner(analysis.user_id, current_user)

    try:
        return service.get_analysis_result(analysis_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis result file not found")


@router.get("/visualization/{analysis_id}")
async def get_visualization(
    analysis_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Download annotated video with pose overlay."""
    result = await db.execute(select(AnalysisDB).where(AnalysisDB.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    _check_owner(analysis.user_id, current_user)

    try:
        video_path = service.get_visualization_path(analysis_id)
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=f"{analysis_id}_annotated.mp4",
            headers={"Content-Disposition": f'attachment; filename="{analysis_id}_annotated.mp4"'},
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Visualization not found")


@router.get("/status/{analysis_id}")
async def get_analysis_status(
    analysis_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Check analysis status."""
    result = await db.execute(select(AnalysisDB).where(AnalysisDB.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if analysis is None:
        return {"analysis_id": analysis_id, "status": "not_found", "exists": False}
    _check_owner(analysis.user_id, current_user)

    return {"analysis_id": analysis_id, "status": analysis.status, "exists": True}


@router.get("/health")
async def health_check():
    """Health check endpoint for analysis service."""
    return {"status": "healthy", "service": "Phase 2A Golf Analysis", "version": "2.0"}
