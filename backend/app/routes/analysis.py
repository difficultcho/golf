"""
Analysis API Routes - Phase 2A

Endpoints for golf swing analysis:
- POST /{video_id} - Run analysis on uploaded video
- GET /result/{analysis_id} - Get analysis results
- GET /visualization/{analysis_id} - Download annotated video
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import json

from app.services.golf_analysis_service import GolfAnalysisService
from app.models.video import AnalysisResult
from app.config import settings


router = APIRouter(prefix="/api/analysis", tags=["analysis"])
service = GolfAnalysisService()


@router.post("/{video_id}", response_model=AnalysisResult)
async def analyze_video(video_id: str):
    """
    Run Phase 2A analysis on uploaded video

    This endpoint performs complete golf swing analysis:
    1. 3D pose estimation (MediaPipe)
    2. Physics simulation (MuJoCo)
    3. Biomechanics metrics computation

    Args:
        video_id: UUID of previously uploaded video

    Returns:
        AnalysisResult with:
        - pose_data: 3D keypoints and confidence scores
        - physics_metrics: Club speed, X-Factor, energy efficiency, etc.
        - visualization_url: URL to download annotated video

    Raises:
        404: Video not found
        500: Analysis failed
    """
    try:
        result = await service.analyze_video(video_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during analysis: {str(e)}"
        )


@router.get("/result/{analysis_id}", response_model=AnalysisResult)
async def get_analysis_result(analysis_id: str):
    """
    Retrieve previously computed analysis result

    Args:
        analysis_id: UUID of the analysis

    Returns:
        AnalysisResult with complete analysis data

    Raises:
        404: Analysis result not found
    """
    try:
        result = service.get_analysis_result(analysis_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis result not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving result: {str(e)}"
        )


@router.get("/visualization/{analysis_id}")
async def get_visualization(analysis_id: str):
    """
    Download annotated video with pose overlay

    Args:
        analysis_id: UUID of the analysis

    Returns:
        MP4 video file with skeleton overlay

    Raises:
        404: Visualization not found
    """
    try:
        video_path = service.get_visualization_path(analysis_id)

        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=f"{analysis_id}_annotated.mp4",
            headers={
                "Content-Disposition": f'attachment; filename="{analysis_id}_annotated.mp4"'
            }
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Visualization not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving visualization: {str(e)}"
        )


@router.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Check if analysis exists (simple status check)

    Args:
        analysis_id: UUID of the analysis

    Returns:
        Status information
    """
    exists = service.analysis_exists(analysis_id)

    return {
        "analysis_id": analysis_id,
        "status": "completed" if exists else "not_found",
        "exists": exists
    }


@router.get("/health")
async def health_check():
    """Health check endpoint for analysis service"""
    return {
        "status": "healthy",
        "service": "Phase 2A Golf Analysis",
        "version": "2.0"
    }
