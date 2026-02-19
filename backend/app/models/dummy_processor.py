import torch
import torch.nn as nn
import cv2
import shutil
import time
import subprocess
from pathlib import Path
from typing import Dict


class DummyVideoProcessor(nn.Module):
    """
    Phase 1 Dummy Video Processor

    This processor demonstrates the video processing pipeline:
    - Loads video with OpenCV
    - Adds a simple watermark text
    - Extracts video metadata
    - Saves processed video

    In Phase 2, this will be replaced with real PyTorch models
    for actual video analysis and processing.
    """

    def __init__(self):
        super().__init__()
        # Dummy parameter to make it a proper PyTorch module
        self.version = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def _try_ffmpeg_convert(self, video_path: Path, output_path: Path) -> Dict:
        """Try to convert video using ffmpeg (handles WebM and other formats)"""
        result = subprocess.run([
            'ffmpeg',
            '-i', str(video_path),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-y',
            str(output_path)
        ], capture_output=True)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

        # Extract metadata from the converted output (more reliable than source for WebM)
        probe = subprocess.run([
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams', '-show_format',
            str(output_path)
        ], capture_output=True, text=True)

        import json
        info = json.loads(probe.stdout)
        video_stream = next((s for s in info.get('streams', []) if s['codec_type'] == 'video'), {})
        duration = float(info.get('format', {}).get('duration', 0) or 0)
        fps_parts = video_stream.get('r_frame_rate', '30/1').split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 and float(fps_parts[1]) > 0 else 30.0
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        nb_frames = video_stream.get('nb_frames')
        frame_count = int(nb_frames) if nb_frames and nb_frames != 'N/A' else int(duration * fps)

        return {
            "duration": round(duration, 2),
            "frame_count": frame_count,
            "fps": round(fps, 2),
            "resolution": f"{width}x{height}",
        }

    def _try_opencv_process(self, video_path: Path, output_path: Path) -> Dict:
        """Process video with OpenCV (add watermark) and optionally re-encode with ffmpeg"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_output = output_path.parent / f"temp_{output_path.name}"

        codecs_to_try = ['avc1', 'x264', 'H264', 'mp4v']
        out = None
        for codec in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
            if out.isOpened():
                break
            out.release()
            out = None

        if out is None:
            cap.release()
            raise ValueError("No suitable OpenCV codec found")

        frames_processed = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            text = "Processed"
            position = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, position, font, 1.5, (0, 0, 0), 5)
            cv2.putText(frame, text, position, font, 1.5, (255, 255, 255), 3)
            out.write(frame)
            frames_processed += 1

        cap.release()
        out.release()

        # Try re-encoding with ffmpeg for better compatibility
        try:
            subprocess.run([
                'ffmpeg', '-i', str(temp_output),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-movflags', '+faststart', '-y', str(output_path)
            ], check=True, capture_output=True)
            temp_output.unlink()
        except (subprocess.CalledProcessError, FileNotFoundError):
            temp_output.rename(output_path)

        return {
            "duration": round(frame_count / fps, 2) if fps > 0 else 0,
            "frame_count": frames_processed,
            "fps": round(fps, 2),
            "resolution": f"{width}x{height}",
        }

    def forward(self, video_path: Path, output_path: Path) -> Dict:
        """
        Process video by adding watermark and extracting metadata.
        Falls back gracefully if OpenCV can't handle the format.
        """
        start_time = time.time()

        # Strategy 1: Try OpenCV processing (works for standard MP4/H.264)
        try:
            stats = self._try_opencv_process(video_path, output_path)
            stats["processing_time"] = round(time.time() - start_time, 2)
            stats["model_version"] = f"dummy_v{self.version.item()}"
            return stats
        except ValueError as e:
            print(f"OpenCV processing failed: {e}")

        # Strategy 2: Try ffmpeg conversion (handles WebM, HEVC, etc.)
        try:
            stats = self._try_ffmpeg_convert(video_path, output_path)
            stats["processing_time"] = round(time.time() - start_time, 2)
            stats["model_version"] = f"dummy_v{self.version.item()}"
            print("Successfully processed with ffmpeg")
            return stats
        except (RuntimeError, FileNotFoundError) as e:
            print(f"ffmpeg processing failed: {e}")

        # Strategy 3: Copy raw file as-is (last resort for Phase 1)
        print("Falling back to raw file copy (no processing)")
        shutil.copy2(str(video_path), str(output_path))

        # Return basic metadata from file stats
        file_size = video_path.stat().st_size
        processing_time = round(time.time() - start_time, 2)

        return {
            "duration": 0,
            "frame_count": 0,
            "fps": 0,
            "resolution": "unknown",
            "processing_time": processing_time,
            "model_version": f"dummy_v{self.version.item()}"
        }

    def get_version(self) -> str:
        """Get the processor version"""
        return f"dummy_v{self.version.item()}"
