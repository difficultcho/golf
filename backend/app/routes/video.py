from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from app.services.video_service import VideoService
from app.models.video import VideoUploadResponse, VideoStatus
from app.config import settings
import os


router = APIRouter(prefix="/api/video", tags=["video"])
service = VideoService()


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(video: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a video file for Phase 2A analysis.
    Processing runs in the background (MediaPipe + MuJoCo + Biomechanics).
    Poll /api/video/status/{video_id} to check progress.
    """
    # Validate file format
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = os.path.splitext(video.filename)[1].lower()
    if file_ext not in settings.ALLOWED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video format. Allowed formats: {', '.join(settings.ALLOWED_VIDEO_FORMATS)}"
        )

    # Read and validate file size
    content = await video.read()
    file_size_mb = len(content) / (1024 * 1024)

    if file_size_mb > settings.MAX_VIDEO_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_VIDEO_SIZE_MB}MB"
        )

    # Generate ID and save
    video_id = service.generate_video_id()

    try:
        await service.save_upload(video_id, content, file_ext)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save video: {str(e)}"
        )

    # Run analysis in the background
    background_tasks.add_task(service.process_video_background, video_id)

    return VideoUploadResponse(
        video_id=video_id,
        status="processing",
        message="Video uploaded, analysis started in background"
    )


@router.get("/status/{video_id}", response_model=VideoStatus)
async def get_status(video_id: str):
    """
    Get processing status for a video

    Args:
        video_id: Unique identifier for the video

    Returns:
        VideoStatus with current processing status

    Raises:
        HTTPException 404: Video not found
    """
    status = service.get_status(video_id)

    if status == "not_found":
        raise HTTPException(status_code=404, detail="Video not found")

    messages = {
        "done": "Processing completed successfully",
        "processing": "Video is being analyzed",
        "failed": "Analysis failed",
    }

    return VideoStatus(
        video_id=video_id,
        status=status,
        message=messages.get(status),
        progress=100 if status == "done" else None
    )


@router.get("/result/{video_id}")
async def get_result(video_id: str):
    """
    Download processed video file

    Args:
        video_id: Unique identifier for the video

    Returns:
        FileResponse with processed video

    Raises:
        HTTPException 404: Processed video not found
    """
    file_path = service.get_processed_video_path(video_id)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Processed video not found. Video may still be processing."
        )

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=f"{video_id}_processed.mp4"
    )


@router.get("/data/{video_id}")
async def get_metadata(video_id: str):
    """
    Download video metadata JSON file

    Args:
        video_id: Unique identifier for the video

    Returns:
        FileResponse with JSON metadata

    Raises:
        HTTPException 404: Metadata not found
    """
    file_path = service.get_metadata_path(video_id)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Metadata not found. Video may not have been processed yet."
        )

    return FileResponse(
        path=file_path,
        media_type="application/json",
        filename=f"{video_id}_metadata.json"
    )


@router.delete("/video/{video_id}")
async def delete_video(video_id: str):
    """
    Delete a video and its associated files (optional endpoint for cleanup)

    Args:
        video_id: Unique identifier for the video

    Returns:
        Success message

    Raises:
        HTTPException 404: Video not found
    """
    if not service.video_exists(video_id):
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete all associated files (raw video may have various extensions)
    processed_path = settings.PROCESSED_VIDEOS_DIR / f"{video_id}.mp4"
    metadata_path = settings.METADATA_DIR / f"{video_id}.json"

    deleted_files = []
    # Find raw video with any extension
    for ext in [".mp4", ".webm", ".mov", ".avi"]:
        raw_path = settings.RAW_VIDEOS_DIR / f"{video_id}{ext}"
        if raw_path.exists():
            raw_path.unlink()
            deleted_files.append(raw_path.name)
    for path in [processed_path, metadata_path]:
        if path.exists():
            path.unlink()
            deleted_files.append(path.name)

    return {
        "message": "Video deleted successfully",
        "video_id": video_id,
        "deleted_files": deleted_files
    }
