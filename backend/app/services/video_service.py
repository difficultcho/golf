import uuid
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from app.config import settings
from app.services.golf_analysis_service import GolfAnalysisService


class VideoService:
    """
    Video processing service handling upload, processing, and retrieval.
    Uses Phase 2A analysis pipeline (MediaPipe + MuJoCo + Biomechanics).
    """

    def __init__(self):
        """Initialize the video service and ensure directories exist"""
        self.analysis_service = GolfAnalysisService()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        settings.RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        settings.PROCESSED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)

    def generate_video_id(self) -> str:
        """Generate a unique video ID using UUID4"""
        return str(uuid.uuid4())

    @staticmethod
    def detect_video_format(content: bytes) -> str:
        """Detect actual video format from file header bytes"""
        if len(content) >= 4:
            if content[:4] == b'\x1a\x45\xdf\xa3':
                return ".webm"
        return ".mp4"

    async def save_upload(self, video_id: str, file_content: bytes, original_ext: str = ".mp4") -> Path:
        """Save uploaded video to raw videos directory"""
        detected_ext = self.detect_video_format(file_content)
        ext = detected_ext if detected_ext != ".mp4" else original_ext
        file_path = settings.RAW_VIDEOS_DIR / f"{video_id}{ext}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        return file_path

    def _convert_to_mp4(self, input_path: Path) -> Path:
        """Convert non-MP4 video to MP4 using ffmpeg for OpenCV compatibility"""
        output_path = input_path.with_suffix(".converted.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(input_path), "-c:v", "libx264",
                 "-preset", "fast", "-crf", "23", str(output_path)],
                capture_output=True, check=True, timeout=120
            )
            print(f"Converted {input_path.name} → {output_path.name}")
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise ValueError(f"ffmpeg conversion failed: {e}")

    def _reencode_h264(self, video_path: Path):
        """Re-encode video to H.264 + AAC for WeChat Mini Program compatibility.
        OpenCV on Linux often falls back to mp4v codec which WeChat cannot play."""
        temp_path = video_path.with_suffix(".tmp.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path),
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-c:a", "aac",
                 "-movflags", "+faststart",
                 str(temp_path)],
                capture_output=True, check=True, timeout=300
            )
            temp_path.rename(video_path)
            print(f"Re-encoded {video_path.name} to H.264")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if temp_path.exists():
                temp_path.unlink()
            raise ValueError(f"ffmpeg re-encoding failed: {e}")

    async def process_video(self, video_id: str):
        """
        Run Phase 2A analysis pipeline on uploaded video.

        Performs: MediaPipe pose estimation → MuJoCo simulation → Biomechanics analysis.
        Saves annotated video to processed_videos/ and enriched metadata to metadata/.
        """
        raw_path = self._find_raw_video(video_id)
        if raw_path is None:
            raise FileNotFoundError(f"Raw video not found: {video_id}")

        # Convert WebM/non-MP4 to MP4 for OpenCV compatibility
        video_path = raw_path
        if raw_path.suffix.lower() != ".mp4":
            video_path = self._convert_to_mp4(raw_path)

        # Run full analysis pipeline
        result = await self.analysis_service.analyze_video(video_id, raw_video_path=video_path)

        # Copy annotated video to processed_videos/ for the existing download endpoint
        processed_path = settings.PROCESSED_VIDEOS_DIR / f"{video_id}.mp4"
        annotated_path = self.analysis_service.get_visualization_path(result.analysis_id)
        shutil.copy2(str(annotated_path), str(processed_path))

        # Re-encode to H.264 + AAC so WeChat Mini Program can play it
        self._reencode_h264(processed_path)

        # Build enriched metadata with analysis results
        pose_data = result.pose_data
        metrics = result.physics_metrics
        metadata = {
            "video_id": video_id,
            "analysis_id": result.analysis_id,
            "upload_time": datetime.now().isoformat(),
            "duration": pose_data.timestamps[-1] if pose_data.timestamps else 0,
            "frame_count": len(pose_data.timestamps),
            "fps": round(len(pose_data.timestamps) / pose_data.timestamps[-1], 2) if pose_data.timestamps and pose_data.timestamps[-1] > 0 else 0,
            "resolution": "unknown",
            "processing_time": round(result.processing_time, 2),
            "model_version": "phase2a_v2.0",
            # Analysis metrics
            "club_head_speed_mph": round(metrics.club_head_speed_mph, 1),
            "x_factor": round(metrics.x_factor, 1),
            "balance_score": round(metrics.balance_score, 1),
            "energy_efficiency": round(metrics.energy_efficiency, 3),
            "swing_duration_sec": round(metrics.swing_duration_sec, 2),
            "peak_torques": {k: round(v, 1) for k, v in metrics.peak_torques.items()},
        }

        # Save metadata
        metadata_path = settings.METADATA_DIR / f"{video_id}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    async def process_video_background(self, video_id: str):
        """Background task wrapper with error handling"""
        try:
            await self.process_video(video_id)
        except Exception as e:
            print(f"Analysis failed for {video_id}: {e}")
            error_metadata = {
                "video_id": video_id,
                "status": "failed",
                "error": str(e),
                "upload_time": datetime.now().isoformat()
            }
            metadata_path = settings.METADATA_DIR / f"{video_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(error_metadata, f, indent=2)

    def _find_raw_video(self, video_id: str):
        """Find raw video file regardless of extension"""
        for ext in [".mp4", ".webm", ".mov", ".avi"]:
            path = settings.RAW_VIDEOS_DIR / f"{video_id}{ext}"
            if path.exists():
                return path
        return None

    def get_status(self, video_id: str) -> str:
        """Get processing status for a video"""
        processed_path = settings.PROCESSED_VIDEOS_DIR / f"{video_id}.mp4"
        if processed_path.exists():
            return "done"

        # Check if failed (metadata exists with failed status)
        metadata_path = settings.METADATA_DIR / f"{video_id}.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            if meta.get("status") == "failed":
                return "failed"

        if self._find_raw_video(video_id) is not None:
            return "processing"

        return "not_found"

    def get_processed_video_path(self, video_id: str) -> Path:
        """Get path to processed video file"""
        return settings.PROCESSED_VIDEOS_DIR / f"{video_id}.mp4"

    def get_metadata_path(self, video_id: str) -> Path:
        """Get path to metadata JSON file"""
        return settings.METADATA_DIR / f"{video_id}.json"

    def video_exists(self, video_id: str) -> bool:
        """Check if a video exists (either raw or processed)"""
        processed_path = settings.PROCESSED_VIDEOS_DIR / f"{video_id}.mp4"
        return self._find_raw_video(video_id) is not None or processed_path.exists()
