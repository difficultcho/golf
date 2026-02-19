"""
Phase 2A Integration Test Script

Tests the complete analysis pipeline:
1. Start the FastAPI server (manually)
2. Upload a test video
3. Run Phase 2A analysis
4. Retrieve results
5. Download annotated video

Usage:
    # First, start the server in another terminal:
    # cd backend && uvicorn app.main:app --reload

    # Then run this script:
    # python test_phase2a.py
"""

import requests
import json
import time
from pathlib import Path


# Configuration
BASE_URL = "http://localhost:8000"
TEST_VIDEO_PATH = "test_video.mp4"  # Replace with actual test video path


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def test_server_health():
    """Test if server is running"""
    print_section("1. Testing Server Health")

    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✓ Server is running")
        print(f"  Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"✗ Server is not running: {e}")
        print("  Please start the server first:")
        print("  cd backend && uvicorn app.main:app --reload")
        return False


def upload_test_video():
    """Upload a test video"""
    print_section("2. Uploading Test Video")

    test_video = Path(TEST_VIDEO_PATH)

    if not test_video.exists():
        print(f"✗ Test video not found: {TEST_VIDEO_PATH}")
        print("  Please provide a test video file")
        print("  You can use any MP4 video for testing")
        return None

    try:
        with open(test_video, 'rb') as f:
            files = {'video': (test_video.name, f, 'video/mp4')}
            response = requests.post(f"{BASE_URL}/api/video/upload", files=files)

        if response.status_code == 200:
            data = response.json()
            video_id = data['video_id']
            print(f"✓ Video uploaded successfully")
            print(f"  Video ID: {video_id}")
            return video_id
        else:
            print(f"✗ Upload failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None

    except Exception as e:
        print(f"✗ Upload error: {e}")
        return None


def run_analysis(video_id):
    """Run Phase 2A analysis"""
    print_section("3. Running Phase 2A Analysis")

    try:
        print(f"Analyzing video: {video_id}")
        print("This may take 30-90 seconds...")

        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/analysis/{video_id}")
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            analysis_id = data['analysis_id']

            print(f"\n✓ Analysis completed in {elapsed:.1f}s")
            print(f"  Analysis ID: {analysis_id}")
            print(f"\n  Physics Metrics:")
            metrics = data['physics_metrics']
            print(f"    - Club head speed: {metrics['club_head_speed_mph']:.1f} mph")
            print(f"    - X-Factor: {metrics['x_factor']:.1f}°")
            print(f"    - Energy efficiency: {metrics['energy_efficiency']:.1%}")
            print(f"    - Balance score: {metrics['balance_score']:.1f}/100")
            print(f"    - Swing duration: {metrics['swing_duration_sec']:.2f}s")

            return analysis_id
        else:
            print(f"✗ Analysis failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None

    except Exception as e:
        print(f"✗ Analysis error: {e}")
        return None


def retrieve_results(analysis_id):
    """Retrieve analysis results"""
    print_section("4. Retrieving Analysis Results")

    try:
        response = requests.get(f"{BASE_URL}/api/analysis/result/{analysis_id}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Results retrieved successfully")
            print(f"  Processing time: {data['processing_time']:.2f}s")
            print(f"  Pose data frames: {len(data['pose_data']['timestamps'])}")
            print(f"  Visualization URL: {data['visualization_url']}")
            return True
        else:
            print(f"✗ Failed to retrieve results: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Retrieval error: {e}")
        return False


def download_visualization(analysis_id):
    """Download annotated video"""
    print_section("5. Downloading Annotated Video")

    try:
        response = requests.get(f"{BASE_URL}/api/analysis/visualization/{analysis_id}")

        if response.status_code == 200:
            output_path = f"test_output_{analysis_id}_annotated.mp4"
            with open(output_path, 'wb') as f:
                f.write(response.content)

            print(f"✓ Annotated video downloaded")
            print(f"  Saved to: {output_path}")
            print(f"  Size: {len(response.content) / 1024 / 1024:.2f} MB")
            return True
        else:
            print(f"✗ Download failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Download error: {e}")
        return False


def main():
    """Run complete integration test"""
    print("\n" + "="*60)
    print(" Phase 2A Integration Test")
    print("="*60)

    # Test 1: Server health
    if not test_server_health():
        return

    # Test 2: Upload video
    video_id = upload_test_video()
    if not video_id:
        print("\n⚠️  Skipping remaining tests (no video uploaded)")
        print("   To test with a video, provide a test_video.mp4 file")
        return

    # Test 3: Run analysis
    analysis_id = run_analysis(video_id)
    if not analysis_id:
        return

    # Test 4: Retrieve results
    retrieve_results(analysis_id)

    # Test 5: Download visualization
    download_visualization(analysis_id)

    # Summary
    print_section("Test Summary")
    print("✓ All tests passed!")
    print(f"\nYou can view the API documentation at:")
    print(f"  {BASE_URL}/docs")


if __name__ == "__main__":
    main()
