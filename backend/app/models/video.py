from pydantic import BaseModel
from typing import Optional, Literal, List, Dict
from datetime import datetime


class VideoUploadResponse(BaseModel):
    """Response model for video upload"""
    video_id: str
    status: str
    message: str


class VideoStatus(BaseModel):
    """Model for video processing status"""
    video_id: str
    status: Literal["processing", "done", "failed", "not_found"]
    message: Optional[str] = None
    progress: Optional[int] = None  # 0-100


class VideoMetadata(BaseModel):
    """Model for video metadata (enriched with Phase 2A analysis)"""
    video_id: str
    duration: float
    frame_count: int
    fps: float
    resolution: str
    processing_time: float
    model_version: str
    upload_time: str
    extra_fields: Optional[dict] = {}
    # Phase 2A analysis fields
    analysis_id: Optional[str] = None
    club_head_speed_mph: Optional[float] = None
    x_factor: Optional[float] = None
    balance_score: Optional[float] = None
    energy_efficiency: Optional[float] = None
    swing_duration_sec: Optional[float] = None
    peak_torques: Optional[Dict[str, float]] = None


# Phase 2A Models

class PoseData(BaseModel):
    """3D pose keypoints from video analysis"""
    joints_3d: List[List[List[float]]]  # (T, 17, 3) - Time × Joints × Coordinates
    confidence: List[float]  # Confidence scores for each frame
    timestamps: List[float]  # Timestamp for each frame (seconds)


class PhysicsMetrics(BaseModel):
    """Biomechanics metrics from MuJoCo simulation"""
    club_head_speed_mph: float  # Club head speed at impact (miles per hour)
    swing_duration_sec: float  # Total swing duration (seconds)
    peak_torques: Dict[str, float]  # Peak torques for major joints (Nm)
    energy_efficiency: float  # Energy transfer efficiency (0-1)
    balance_score: float  # Balance/stability score (0-100)
    x_factor: float  # Shoulder-hip separation angle (degrees)


class AnalysisResult(BaseModel):
    """Complete Phase 2A analysis result"""
    video_id: str
    analysis_id: str
    pose_data: PoseData
    physics_metrics: PhysicsMetrics
    processing_time: float  # Total analysis time (seconds)
    visualization_url: Optional[str] = None  # URL to annotated video
