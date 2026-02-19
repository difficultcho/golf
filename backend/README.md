# Video Processing Backend

FastAPI backend service for WeChat Mini Program video processing with PyTorch.

## Setup

### 1. Create Virtual Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Development Server

```bash
# Method 1: Using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Method 2: Using Python
python -m app.main

# Method 3: Using the main.py directly
python app/main.py
```

The server will start at `http://localhost:8000`

## API Documentation

Once the server is running, access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Upload Video
```
POST /api/video/upload
Content-Type: multipart/form-data

Body:
- video: <file>

Response:
{
  "video_id": "uuid",
  "status": "processing",
  "message": "Video uploaded and processing started"
}
```

### 2. Get Status
```
GET /api/video/status/{video_id}

Response:
{
  "video_id": "uuid",
  "status": "done",
  "message": "Processing completed successfully",
  "progress": 100
}
```

### 3. Download Processed Video
```
GET /api/video/result/{video_id}

Returns: video/mp4 file
```

### 4. Download Metadata
```
GET /api/video/data/{video_id}

Returns: JSON file with video metadata
```

## Testing

### Using curl

```bash
# Upload video
curl -X POST -F "video=@test_video.mp4" http://localhost:8000/api/video/upload

# Get status (replace VIDEO_ID)
curl http://localhost:8000/api/video/status/VIDEO_ID

# Download result (in browser or save to file)
curl http://localhost:8000/api/video/result/VIDEO_ID -o processed.mp4

# Download metadata
curl http://localhost:8000/api/video/data/VIDEO_ID -o metadata.json
```

### Using Python requests

```python
import requests

# Upload
with open('test_video.mp4', 'rb') as f:
    response = requests.post('http://localhost:8000/api/video/upload',
                            files={'video': f})
video_id = response.json()['video_id']

# Check status
status = requests.get(f'http://localhost:8000/api/video/status/{video_id}')
print(status.json())

# Download processed video
result = requests.get(f'http://localhost:8000/api/video/result/{video_id}')
with open('processed.mp4', 'wb') as f:
    f.write(result.content)
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   ├── video.py         # Pydantic models
│   │   └── dummy_processor.py  # PyTorch processor
│   ├── routes/
│   │   └── video.py         # API endpoints
│   └── services/
│       └── video_service.py # Business logic
├── data/
│   ├── raw_videos/          # Uploaded videos
│   ├── processed_videos/    # Processed outputs
│   └── metadata/            # JSON metadata files
└── requirements.txt
```

## Configuration

Create a `.env` file in the backend directory to override defaults:

```env
APP_NAME=Video Processing API
HOST=0.0.0.0
PORT=8000
DEBUG=True
MAX_VIDEO_SIZE_MB=100
DUMMY_PROCESSING_DELAY=1.0
```

## Phase 1 vs Phase 2

**Phase 1 (Current):**
- Dummy video processing (adds watermark)
- Synchronous processing
- Local file storage
- No authentication

**Phase 2 (Future):**
- Real PyTorch models
- Async processing with Celery + Redis
- Cloud storage (S3/OSS)
- User authentication
- WebSocket for real-time updates

## Notes

- Maximum video size: 100MB (configurable)
- Supported formats: .mp4, .mov, .avi
- Processing is synchronous in Phase 1
- CORS is wide open for development (restrict in production)
