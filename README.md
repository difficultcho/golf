# AI-Driven Golf Swing Analysis System

**WeChat Mini Program + PyTorch + MuJoCo Physics Simulation**

An intelligent golf swing analysis system that combines computer vision, robotics simulation, and reinforcement learning to provide actionable swing improvement suggestions.

## Project Overview

**Phase 1 (✅ Implemented)**: Video processing pipeline
1. Record golf swing videos in WeChat Mini Program
2. Upload videos to FastAPI backend
3. Process videos with PyTorch (dummy watermark processor)
4. View and download processed videos with metadata

**Phase 2 (🚧 Planned)**: AI + Physics analysis
1. **3D Pose Estimation**: Extract 3D joint trajectories from monocular video
2. **MuJoCo Simulation**: Reconstruct swing in physics simulator
3. **Biomechanics Analysis**: Compute joint torques, energy transfer, GRF
4. **RL Optimization**: Generate personalized improvement suggestions
5. **Visual Feedback**: Annotated videos, 3D animations, physics charts

## Project Structure

```
golf/
├── backend/                 # Python backend
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── config.py       # Configuration
│   │   ├── models/         # Pydantic models & PyTorch processor
│   │   ├── routes/         # API endpoints
│   │   └── services/       # Business logic
│   ├── data/               # Video storage
│   │   ├── raw_videos/
│   │   ├── processed_videos/
│   │   └── metadata/
│   └── requirements.txt
│
├── pages/                   # WeChat Mini Program pages
│   ├── index/              # Home page
│   ├── record/             # Video recording
│   ├── upload/             # Upload & processing
│   └── result/             # View results
│
├── components/             # Reusable components
│   └── navigation-bar/
│
├── utils/                  # Shared utilities
│   └── config.js          # API configuration
│
├── app.js                  # Mini Program entry
├── app.json                # Mini Program config
└── prd.md                  # Product requirements

## Quick Start

### 1. Setup Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 2. Setup WeChat Mini Program

1. Open WeChat DevTools (微信开发者工具)
2. Import the project (select the `golf` directory)
3. In Settings → Project Settings:
   - Enable "Do not verify valid domain name" (不校验合法域名)
   - This allows localhost connections during development
4. Compile and run in simulator

## Usage Flow

1. **Home Page**: Click "开始录制" (Start Recording)
2. **Record Page**:
   - Grant camera permission
   - Record video (max 60 seconds)
   - Click "上传视频" (Upload Video)
3. **Upload Page**:
   - Watch upload progress
   - Wait for processing to complete
   - Click "查看结果" (View Result)
4. **Result Page**:
   - Play processed video (with "Processed" watermark)
   - View metadata (duration, frames, resolution)
   - Save video to album or view JSON data

## API Endpoints

### POST /api/video/upload
Upload a video file
- Request: multipart/form-data with `video` file
- Response: `{ video_id, status, message }`

### GET /api/video/status/{video_id}
Check processing status
- Response: `{ video_id, status, message, progress }`
- Status values: "processing", "done", "failed"

### GET /api/video/result/{video_id}
Download processed video
- Returns: video/mp4 file

### GET /api/video/data/{video_id}
Download metadata JSON
- Returns: JSON with video metadata

## Testing

### Backend Testing

```bash
# Create a test video (or use any video file)
# Test upload
curl -X POST -F "video=@test.mp4" http://localhost:8000/api/video/upload

# Copy the video_id from response, then:
# Test status
curl http://localhost:8000/api/video/status/{video_id}

# Test download (in browser or save)
curl http://localhost:8000/api/video/result/{video_id} -o processed.mp4
curl http://localhost:8000/api/video/data/{video_id} -o metadata.json
```

### Frontend Testing

1. **Simulator Testing**:
   - Test in WeChat DevTools simulator
   - Test all pages and navigation
   - Test error handling (disable backend)

2. **Real Device Testing**:
   - Scan QR code in WeChat DevTools
   - Test camera recording on real device
   - Test video save to album

## Configuration

### Backend Configuration

Edit `backend/app/config.py` or create `backend/.env`:

```env
DEBUG=True
HOST=0.0.0.0
PORT=8000
MAX_VIDEO_SIZE_MB=100
DUMMY_PROCESSING_DELAY=1.0
```

### Frontend Configuration

Edit `utils/config.js`:

```javascript
const ENV = 'development'  // or 'production'

const config = {
  development: {
    API_BASE: 'http://localhost:8000'
  },
  production: {
    API_BASE: 'https://your-domain.com'
  }
}
```

## Features

### Phase 1 (Current)

- ✅ Video recording with camera component
- ✅ Real-time upload progress tracking
- ✅ Dummy PyTorch video processing (watermark)
- ✅ Video playback and download
- ✅ Metadata extraction and display
- ✅ Error handling and retries
- ✅ Permission management (camera, album)

### Phase 2 (Planned) - AI-Driven Golf Swing Physics Analysis

**Core Technologies**: PyTorch + MuJoCo + Reinforcement Learning

#### 🎯 Key Features

**AI Video Analysis Module**:
- ⏳ 3D human pose estimation (MediaPipe)
- ⏳ Golf club trajectory tracking (YOLOv8)
- ⏳ Swing phase segmentation (Address → Backswing → Downswing → Impact → Follow-through)
- ⏳ Multi-task learning network

**MuJoCo Physics Simulation Module**:
- ⏳ Custom 26-DOF humanoid + golf club MJCF model
- ⏳ Trajectory-driven simulation (mocap control)
- ⏳ Inverse dynamics analysis (joint torques via mj_inverse)
- ⏳ Contact force analysis (ground reaction forces)
- ⏳ Kinetic chain energy transfer analysis
- ⏳ Impact moment physics prediction

**Reinforcement Learning Optimization** (Advanced):
- ⏳ PPO/SAC policy training in MuJoCo environment
- ⏳ Multi-objective reward function design
- ⏳ Personalized swing optimization suggestions
- ⏳ User vs RL-optimal trajectory comparison

#### 📊 Analysis Outputs

- Club head speed curves
- Joint torque heatmaps
- Energy efficiency scoring (legs → torso → arms → club)
- Balance stability analysis (center of pressure)
- X-Factor (shoulder-hip separation angle)
- Ball flight prediction (distance, direction)

#### 🔧 Infrastructure

- ⏳ Async processing with Celery + Redis
- ⏳ Cloud storage (S3/OSS)
- ⏳ User authentication (WeChat login)
- ⏳ WebSocket for real-time progress updates

## Technical Stack

**Frontend:**
- WeChat Mini Program
- Skyline Renderer + glass-easel
- Custom navigation component
- ECharts (data visualization - Phase 2)

**Backend (Phase 1):**
- Python 3.9+
- FastAPI (Web framework)
- PyTorch 2.1.2 (Deep learning)
- OpenCV (Video I/O)
- FFmpeg (Video encoding)
- Uvicorn (ASGI server)

**Backend (Phase 2 - Planned):**
- **MuJoCo 3.0+** (Physics simulation)
- **MediaPipe** (3D pose estimation)
- **YOLOv8** (Object detection & tracking)
- **Stable-Baselines3** (Reinforcement learning)
- **Gymnasium** (RL environment interface)
- Celery + Redis (Async task queue)
- NumPy/SciPy (Scientific computing)

## Development Notes

### WeChat Mini Program

- Uses Skyline renderer (modern rendering engine)
- Custom navigation bar on all pages
- Requires camera and album permissions
- Video paths are temporary (upload immediately)

### Backend

- Synchronous processing in Phase 1
- CORS enabled for WeChat mini program
- Files stored in local filesystem
- No authentication in Phase 1
- UUID-based video IDs

## Troubleshooting

### Backend Issues

**Problem**: Dependencies installation fails
- Solution: Ensure Python 3.9+ is installed
- For macOS M1/M2: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Problem**: Video processing fails
- Solution: Check video format (mp4, mov, avi supported)
- Check file size (max 100MB)

### Frontend Issues

**Problem**: Camera not working
- Solution: Grant camera permission in simulator settings
- Real device: Check WeChat permissions in phone settings

**Problem**: Upload fails with network error
- Solution: Ensure backend is running on `localhost:8000`
- Enable "Do not verify domain name" in WeChat DevTools

**Problem**: Video playback fails
- Solution: Check that video processing completed
- Verify video URL is accessible

## Security Notes

⚠️ **Phase 1 is for development/testing only**

- No user authentication
- CORS allows all origins
- Anyone with video_id can access videos
- Local file storage (not scalable)

Add authentication and proper security in Phase 2 before production.

## Documentation

- [PRD (Product Requirements)](PRD.md) - Complete project requirements including Phase 2 planning
- [CLAUDE.md](CLAUDE.md) - Development guide for Claude Code
- [Backend README](backend/README.md) - Backend implementation details
- [MuJoCo Model Design](docs/MUJOCO_MODEL.md) - Humanoid + golf club MJCF model specifications
- [Technical Architecture](docs/ARCHITECTURE.md) - System architecture and data flow

## License

MIT

## Support

For issues or questions, refer to the PRD or implementation plan.
