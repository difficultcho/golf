"""
GolfAnalysisService - Phase 2A Analysis Pipeline Orchestration

This service coordinates the complete analysis workflow:
1. Pose estimation (MediaPipe)
2. Physics simulation (MuJoCo)
3. Biomechanics analysis
4. Result aggregation
"""

import uuid
import json
import time
from pathlib import Path
from datetime import datetime

from app.config import settings
from app.models.pose_estimator_simple import PoseEstimatorSimple
from app.models.mujoco_simulator import MuJoCoSimulator
from app.models.physics_analyzer import PhysicsAnalyzer
from app.models.video import PoseData, PhysicsMetrics, AnalysisResult


class GolfAnalysisService:
    """
    Orchestrates Phase 2A analysis pipeline

    Coordinates pose estimation, physics simulation, and metrics computation
    to produce comprehensive golf swing analysis.
    """

    def __init__(self):
        """Initialize analysis pipeline components"""
        # Initialize processors
        self.pose_estimator = PoseEstimatorSimple()

        # MuJoCo simulator with model path from config
        mjcf_path = settings.BASE_DIR / settings.MJCF_MODEL_PATH
        self.simulator = MuJoCoSimulator(str(mjcf_path))

        self.analyzer = PhysicsAnalyzer()

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create Phase 2A data directories if they don't exist"""
        settings.POSE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        settings.VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
        print("Phase 2A directories initialized:")
        print(f"  - Pose data: {settings.POSE_DATA_DIR}")
        print(f"  - Analysis results: {settings.ANALYSIS_RESULTS_DIR}")
        print(f"  - Visualizations: {settings.VISUALIZATION_DIR}")

    def generate_analysis_id(self) -> str:
        """Generate unique analysis ID using UUID4"""
        return str(uuid.uuid4())

    def _find_raw_video(self, video_id: str) -> Path:
        """Find raw video file regardless of extension"""
        for ext in [".mp4", ".webm", ".mov", ".avi"]:
            path = settings.RAW_VIDEOS_DIR / f"{video_id}{ext}"
            if path.exists():
                return path
        return None

    async def analyze_video(self, video_id: str, raw_video_path: Path = None) -> AnalysisResult:
        """
        Run complete Phase 2A analysis on uploaded video

        Args:
            video_id: UUID of the uploaded video
            raw_video_path: Optional explicit path to raw video file

        Returns:
            AnalysisResult with pose data, physics metrics, and visualization URL

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If analysis fails
        """
        analysis_id = self.generate_analysis_id()
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting Phase 2A Analysis")
        print(f"Video ID: {video_id}")
        print(f"Analysis ID: {analysis_id}")
        print(f"{'='*60}\n")

        # 1. Verify raw video exists
        raw_path = raw_video_path or self._find_raw_video(video_id)
        if raw_path is None or not raw_path.exists():
            raise FileNotFoundError(f"Video {video_id} not found")

        print(f"✓ Raw video found: {raw_path}")

        # 2. Stage 1: Pose Estimation
        print("\n[Stage 1/3] Running pose estimation...")
        pose_output = settings.POSE_DATA_DIR / f"{analysis_id}_pose.json"
        annotated_video = settings.VISUALIZATION_DIR / f"{analysis_id}_annotated.mp4"

        try:
            pose_result = self.pose_estimator.process_video(raw_path, annotated_video)
            print(f"  ✓ Extracted {len(pose_result['joints_3d'])} frames")
            print(f"  ✓ Average confidence: {sum(pose_result['confidence'])/len(pose_result['confidence']):.2f}")
            print(f"  ✓ Annotated video saved: {annotated_video}")
        except Exception as e:
            raise ValueError(f"Pose estimation failed: {str(e)}")

        # Save pose data
        with open(pose_output, 'w') as f:
            json.dump(pose_result, f, indent=2)
        print(f"  ✓ Pose data saved: {pose_output}")

        # 3. Stage 2: MuJoCo Simulation
        print("\n[Stage 2/3] Running MuJoCo physics simulation...")
        sim_output = settings.ANALYSIS_RESULTS_DIR / f"{analysis_id}_physics.json"

        try:
            sim_result = self.simulator(pose_output, sim_output)
            print(f"  ✓ Simulated {sim_result['frame_count']} frames")
            print(f"  ✓ Physics data saved: {sim_output}")
        except Exception as e:
            raise ValueError(f"Physics simulation failed: {str(e)}")

        # 4. Stage 3: Biomechanics Analysis
        print("\n[Stage 3/3] Computing biomechanics metrics...")
        metrics_output = settings.ANALYSIS_RESULTS_DIR / f"{analysis_id}_metrics.json"

        try:
            metrics_result = self.analyzer(sim_output, metrics_output)
            print(f"  ✓ Club head speed: {metrics_result['club_head_speed_mph']:.1f} mph")
            print(f"  ✓ X-Factor: {metrics_result['x_factor']:.1f}°")
            print(f"  ✓ Energy efficiency: {metrics_result['energy_efficiency']:.1%}")
            print(f"  ✓ Balance score: {metrics_result['balance_score']:.1f}/100")
            print(f"  ✓ Metrics saved: {metrics_output}")
        except Exception as e:
            raise ValueError(f"Biomechanics analysis failed: {str(e)}")

        # 5. Create comprehensive result object
        processing_time = time.time() - start_time

        # Create Pydantic models
        pose_data = PoseData(
            joints_3d=pose_result['joints_3d'],
            confidence=pose_result['confidence'],
            timestamps=pose_result['timestamps']
        )

        physics_metrics = PhysicsMetrics(
            club_head_speed_mph=metrics_result['club_head_speed_mph'],
            swing_duration_sec=metrics_result['swing_duration_sec'],
            peak_torques=metrics_result['peak_torques'],
            energy_efficiency=metrics_result['energy_efficiency'],
            balance_score=metrics_result['balance_score'],
            x_factor=metrics_result['x_factor']
        )

        result = AnalysisResult(
            video_id=video_id,
            analysis_id=analysis_id,
            pose_data=pose_data,
            physics_metrics=physics_metrics,
            processing_time=processing_time,
            visualization_url=f"/api/analysis/visualization/{analysis_id}"
        )

        # Save complete result
        result_path = settings.ANALYSIS_RESULTS_DIR / f"{analysis_id}_result.json"
        with open(result_path, 'w') as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"✓ Analysis Complete!")
        print(f"  Total time: {processing_time:.2f}s")
        print(f"  Result saved: {result_path}")
        print(f"{'='*60}\n")

        return result

    def get_analysis_result(self, analysis_id: str) -> AnalysisResult:
        """
        Retrieve previously computed analysis result

        Args:
            analysis_id: UUID of the analysis

        Returns:
            AnalysisResult object

        Raises:
            FileNotFoundError: If analysis result doesn't exist
        """
        result_path = settings.ANALYSIS_RESULTS_DIR / f"{analysis_id}_result.json"

        if not result_path.exists():
            raise FileNotFoundError(f"Analysis result not found: {analysis_id}")

        with open(result_path, 'r') as f:
            data = json.load(f)

        return AnalysisResult(**data)

    def get_visualization_path(self, analysis_id: str) -> Path:
        """
        Get path to annotated video visualization

        Args:
            analysis_id: UUID of the analysis

        Returns:
            Path to visualization video

        Raises:
            FileNotFoundError: If visualization doesn't exist
        """
        video_path = settings.VISUALIZATION_DIR / f"{analysis_id}_annotated.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Visualization not found: {analysis_id}")

        return video_path

    def analysis_exists(self, analysis_id: str) -> bool:
        """Check if analysis result exists"""
        result_path = settings.ANALYSIS_RESULTS_DIR / f"{analysis_id}_result.json"
        return result_path.exists()
