"""
PoseEstimator - Pose Estimation using MediaPipe Tasks API

Uses MediaPipe PoseLandmarker (Tasks API) compatible with mediapipe >= 0.10.14.
Requires a pose_landmarker.task model file in assets/.
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed. Install with: pip install mediapipe")

# MediaPipe 33 landmarks -> COCO 17 keypoints mapping
# COCO index: MediaPipe index
MEDIAPIPE_TO_COCO = [
    0,   # 0  nose
    2,   # 1  left_eye
    5,   # 2  right_eye
    7,   # 3  left_ear
    8,   # 4  right_ear
    11,  # 5  left_shoulder
    12,  # 6  right_shoulder
    13,  # 7  left_elbow
    14,  # 8  right_elbow
    15,  # 9  left_wrist
    16,  # 10 right_wrist
    23,  # 11 left_hip
    24,  # 12 right_hip
    25,  # 13 left_knee
    26,  # 14 right_knee
    27,  # 15 left_ankle
    28,  # 16 right_ankle
]

# COCO 17 skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6),                                  # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),        # Arms
    (5, 11), (6, 12), (11, 12),             # Torso
    (11, 13), (13, 15), (12, 14), (14, 16), # Legs
]


class PoseEstimatorSimple:
    """
    Pose estimator using MediaPipe Tasks PoseLandmarker API.

    Compatible with mediapipe >= 0.10.14. Requires pose_landmarker.task
    model file (download from Google MediaPipe model zoo).
    """

    def __init__(self, device: str = 'cpu', model_path: str = None):
        self.device = device

        # COCO 17 keypoint names
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        self.landmarker = None
        self.mediapipe_ready = False

        if MEDIAPIPE_AVAILABLE:
            try:
                self._init_landmarker(model_path)
            except Exception as e:
                print(f"Warning: Failed to initialize MediaPipe: {e}")

    def _init_landmarker(self, model_path: str):
        """Initialize PoseLandmarker with Tasks API"""
        if model_path is None:
            from app.config import settings
            model_path = str(settings.BASE_DIR / "assets" / "pose_landmarker.task")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Pose landmarker model not found: {model_path}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task\n"
                "Save as: backend/assets/pose_landmarker.task"
            )

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        print("✓ MediaPipe PoseLandmarker initialized")
        self.mediapipe_ready = True

    def process_video(self, video_path: Path, output_path: Path) -> Dict:
        """
        Extract pose from video and generate annotated output.

        Args:
            video_path: Path to input video
            output_path: Path to save annotated video

        Returns:
            Dictionary with pose data
        """
        if not MEDIAPIPE_AVAILABLE or not self.mediapipe_ready:
            return self._generate_dummy_pose_data(video_path, output_path)

        # Re-initialize landmarker for each video to reset internal timestamp state.
        # MediaPipe VIDEO mode requires strictly increasing timestamps; reusing a
        # landmarker across videos causes the second video (starting at t=0) to be
        # silently rejected, returning no detections.
        if self.landmarker is not None:
            self.landmarker.close()
        self._init_landmarker(None)

        start_time = time.time()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer with codec fallback
        out = None
        for codec in ['avc1', 'x264', 'H264', 'mp4v']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if out.isOpened():
                break
            out.release()
        if out is None or not out.isOpened():
            raise RuntimeError("No suitable video codec found")

        all_joints_2d = []
        all_confidence = []
        all_timestamps = []

        frame_idx = 0
        print(f"Processing {total_frames} frames with MediaPipe...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            all_timestamps.append(timestamp)
            timestamp_ms = int(frame_idx * 1000.0 / fps)

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    landmarks = result.pose_landmarks[0]  # First detected person

                    joints_3d = []
                    visibilities = []
                    for mp_idx in MEDIAPIPE_TO_COCO:
                        lm = landmarks[mp_idx]
                        joints_3d.append([float(lm.x), float(lm.y), float(lm.z)])
                        vis = float(lm.visibility) if (hasattr(lm, 'visibility') and lm.visibility is not None) else 1.0
                        visibilities.append(vis)

                    all_joints_2d.append(joints_3d)
                    all_confidence.append(sum(visibilities) / len(visibilities))

                    # Draw skeleton on frame
                    for start_idx, end_idx in COCO_SKELETON:
                        p1 = joints_3d[start_idx]
                        p2 = joints_3d[end_idx]
                        x1, y1 = int(p1[0] * width), int(p1[1] * height)
                        x2, y2 = int(p2[0] * width), int(p2[1] * height)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    for mp_idx in MEDIAPIPE_TO_COCO:
                        lm = landmarks[mp_idx]
                        x, y = int(lm.x * width), int(lm.y * height)
                        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                else:
                    all_joints_2d.append([[0.0, 0.0, 0.0]] * 17)
                    all_confidence.append(0.0)

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                all_joints_2d.append([[0.0, 0.0, 0.0]] * 17)
                all_confidence.append(0.0)

            out.write(frame)
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")

        cap.release()
        out.release()

        processing_time = time.time() - start_time

        return {
            "joints_3d": all_joints_2d,
            "confidence": all_confidence,
            "timestamps": all_timestamps,
            "duration": float(total_frames / fps) if fps > 0 else 0.0,
            "frame_count": frame_idx,
            "fps": float(fps),
            "resolution": f"{width}x{height}",
            "processing_time": processing_time,
            "model_version": "mediapipe_pose_landmarker_v1.0"
        }

    def _generate_dummy_pose_data(self, video_path: Path, output_path: Path) -> Dict:
        """Generate dummy pose data when MediaPipe is not available"""
        print("Generating dummy pose data (MediaPipe not available)...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        for codec in ['avc1', 'x264', 'H264', 'mp4v']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if out.isOpened():
                break
            out.release()
        if out is None or not out.isOpened():
            raise RuntimeError("No suitable video codec found")

        # Dummy T-pose keypoints
        dummy_keypoints = [
            [0.5, 0.15, 0.0], [0.48, 0.13, 0.0], [0.52, 0.13, 0.0],
            [0.46, 0.15, 0.0], [0.54, 0.15, 0.0], [0.35, 0.3, 0.0],
            [0.65, 0.3, 0.0], [0.25, 0.45, 0.0], [0.75, 0.45, 0.0],
            [0.2, 0.6, 0.0], [0.8, 0.6, 0.0], [0.4, 0.55, 0.0],
            [0.6, 0.55, 0.0], [0.38, 0.75, 0.0], [0.62, 0.75, 0.0],
            [0.37, 0.95, 0.0], [0.63, 0.95, 0.0],
        ]

        all_joints_3d = []
        all_confidence = []
        all_timestamps = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps
            all_timestamps.append(timestamp)
            all_joints_3d.append(dummy_keypoints)
            all_confidence.append(1.0)
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        return {
            "joints_3d": all_joints_3d,
            "confidence": all_confidence,
            "timestamps": all_timestamps,
            "duration": float(total_frames / fps) if fps > 0 else 0.0,
            "frame_count": frame_idx,
            "fps": float(fps),
            "resolution": f"{width}x{height}",
            "processing_time": 0.5,
            "model_version": "dummy_pose_v1.0"
        }

    def close(self):
        """Release MediaPipe resources"""
        if self.landmarker is not None:
            self.landmarker.close()

    def get_version(self) -> str:
        """Get the model version"""
        return "mediapipe_pose_landmarker_v1.0" if self.mediapipe_ready else "dummy_pose_v1.0"
