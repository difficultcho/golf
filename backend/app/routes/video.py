from typing import Annotated

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models.db_models import User, Video as VideoDB, UserRole
from app.models.video import VideoUploadResponse, VideoStatus
from app.services.video_service import VideoService
import os


router = APIRouter(prefix="/api/video", tags=["video"])
service = VideoService()


def _check_video_owner(video: VideoDB, user: User):
    """Raise 404 if user doesn't own the video (admins bypass)."""
    if user.role != UserRole.admin and video.user_id != user.id:
        raise HTTPException(status_code=404, detail="Video not found")


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    video: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """Upload a video file for Phase 2A analysis."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = os.path.splitext(video.filename)[1].lower()
    if file_ext not in settings.ALLOWED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video format. Allowed: {', '.join(settings.ALLOWED_VIDEO_FORMATS)}"
        )

    content = await video.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > settings.MAX_VIDEO_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large. Max: {settings.MAX_VIDEO_SIZE_MB}MB")

    video_id = service.generate_video_id()

    try:
        await service.save_upload(video_id, content, file_ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    # Create DB record
    video_record = VideoDB(id=video_id, user_id=current_user.id, filename=video.filename, status="processing")
    db.add(video_record)
    await db.commit()

    background_tasks.add_task(service.process_video_background, video_id)

    return VideoUploadResponse(
        video_id=video_id,
        status="processing",
        message="Video uploaded, analysis started in background"
    )


@router.get("/status/{video_id}", response_model=VideoStatus)
async def get_status(
    video_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get processing status for a video."""
    result = await db.execute(select(VideoDB).where(VideoDB.id == video_id))
    video = result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    _check_video_owner(video, current_user)

    status = service.get_status(video_id)
    messages = {
        "done": "Processing completed successfully",
        "processing": "Video is being analyzed",
        "failed": "Analysis failed",
    }
    return VideoStatus(
        video_id=video_id,
        status=status,
        message=messages.get(status),
        progress=100 if status == "done" else None,
    )


@router.get("/result/{video_id}")
async def get_result(
    video_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Download processed video file."""
    result = await db.execute(select(VideoDB).where(VideoDB.id == video_id))
    video = result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    _check_video_owner(video, current_user)

    file_path = service.get_processed_video_path(video_id)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found. May still be processing.")

    return FileResponse(path=file_path, media_type="video/mp4", filename=f"{video_id}_processed.mp4")


@router.get("/data/{video_id}")
async def get_metadata(
    video_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Download video metadata JSON file."""
    result = await db.execute(select(VideoDB).where(VideoDB.id == video_id))
    video = result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    _check_video_owner(video, current_user)

    file_path = service.get_metadata_path(video_id)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Metadata not found.")

    return FileResponse(path=file_path, media_type="application/json", filename=f"{video_id}_metadata.json")


@router.delete("/video/{video_id}")
async def delete_video(
    video_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Delete a video and its associated files."""
    result = await db.execute(select(VideoDB).where(VideoDB.id == video_id))
    video = result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    _check_video_owner(video, current_user)

    # Delete files
    deleted_files = []
    for ext in [".mp4", ".webm", ".mov", ".avi"]:
        raw_path = settings.RAW_VIDEOS_DIR / f"{video_id}{ext}"
        if raw_path.exists():
            raw_path.unlink()
            deleted_files.append(raw_path.name)
    for path in [
        settings.PROCESSED_VIDEOS_DIR / f"{video_id}.mp4",
        settings.METADATA_DIR / f"{video_id}.json",
    ]:
        if path.exists():
            path.unlink()
            deleted_files.append(path.name)

    # Delete DB record (cascades to analyses)
    await db.delete(video)
    await db.commit()

    return {"message": "Video deleted successfully", "video_id": video_id, "deleted_files": deleted_files}
