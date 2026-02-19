# Technical Architecture

## System Overview

This document describes the technical architecture of the AI-Driven Golf Swing Analysis System, covering both Phase 1 (implemented) and Phase 2 (planned).

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     WeChat Mini Program                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Record  │→ │  Upload  │→ │Processing│→ │  Result  │       │
│  │   Page   │  │   Page   │  │   Page   │  │   Page   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│         │              │              ↓              ↑           │
└─────────┼──────────────┼──────────────┼──────────────┼──────────┘
          │              │              │              │
          │   HTTP/HTTPS │              │ WebSocket    │
          │              │              │ (Phase 2)    │
          ↓              ↓              ↓              │
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Layer (routes/)                    │  │
│  │   /video/upload  │  /video/status  │  /analysis/result   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Service Layer (services/)                │  │
│  │  VideoService  │  GolfAnalysisService  │  VisualizationSvc│  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Model Layer (models/)                    │  │
│  │  ┌───────────┐  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │  Phase 1  │  │   Phase 2A   │  │    Phase 2B      │  │  │
│  │  │  Dummy    │  │  AI Vision   │  │  RL Optimizer    │  │  │
│  │  │Processor  │  │   + MuJoCo   │  │                  │  │  │
│  │  └───────────┘  └──────────────┘  └──────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Storage & Processing Engine                  │  │
│  │  File System  │  Celery Queue  │  Redis Cache  │  GPU    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Video Processing Pipeline (✅ Implemented)

### 1.1 Frontend Flow

```
User Action → Camera API → Video Recording → Temp File
                                                  ↓
                                          wx.uploadFile()
                                                  ↓
                                          Backend /upload
                                                  ↓
                                    Poll /status/{video_id}
                                                  ↓
                                    Download /result/{video_id}
```

### 1.2 Backend Processing

```python
# Video upload → Processing → Storage
POST /api/video/upload
    ↓
VideoService.save_upload(video_id, content)
    → Save to: data/raw_videos/{video_id}.mp4
    ↓
VideoService.process_video(video_id)
    ↓
DummyVideoProcessor.forward(input_path, output_path)
    → OpenCV: Add watermark
    → FFmpeg: Re-encode to H.264+AAC
    → Save to: data/processed_videos/{video_id}.mp4
    ↓
Generate metadata JSON
    → Save to: data/metadata/{video_id}.json
    ↓
Return: {video_id, status: "done"}
```

### 1.3 Data Flow

```
┌─────────────┐
│  Raw Video  │ → Upload → data/raw_videos/{id}.mp4
└─────────────┘
       ↓
┌─────────────┐
│  Processing │ → OpenCV + FFmpeg
└─────────────┘
       ↓
┌─────────────────┐
│ Processed Video │ → data/processed_videos/{id}.mp4
└─────────────────┘
       ↓
┌─────────────┐
│  Metadata   │ → data/metadata/{id}.json
└─────────────┘
```

### 1.4 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI | REST API endpoints |
| Video I/O | OpenCV | Frame reading/writing |
| Codec | FFmpeg | H.264+AAC encoding |
| DL Framework | PyTorch | Model foundation (dummy in Phase 1) |
| Serialization | Pydantic | Data validation |

---

## Phase 2: AI + Physics Analysis Pipeline (🚧 Planned)

### 2.1 Analysis Pipeline

```
Video Upload
    ↓
┌─────────────────────────────────────────────────────┐
│ Stage 1: AI Video Analysis (30s)                    │
├─────────────────────────────────────────────────────┤
│ 1. 2D Pose Detection (MediaPipe)                     │
│    → Detect 17 keypoints per frame                  │
│                                                      │
│ 2. 3D Pose Lifting (VideoPose3D)                    │
│    → 2D → 3D trajectory                             │
│    → Output: (T, 17, 3) joint positions             │
│                                                      │
│ 3. Object Detection (YOLOv8)                        │
│    → Track golf club trajectory                     │
│    → Track golf ball (if visible)                   │
│                                                      │
│ 4. Temporal Segmentation                            │
│    → Identify swing phases:                         │
│      - Address: [0, t1]                             │
│      - Backswing: [t1, t2]                          │
│      - Downswing: [t2, t3]                          │
│      - Impact: [t3, t4]                             │
│      - Follow-through: [t4, T]                      │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Stage 2: MuJoCo Physics Simulation (20s)            │
├─────────────────────────────────────────────────────┤
│ 1. Load Model                                        │
│    → humanoid_golf.xml (26 DOF + club)              │
│                                                      │
│ 2. Trajectory Smoothing                             │
│    → Savitzky-Golay filter                          │
│    → Remove jitter from AI predictions              │
│                                                      │
│ 3. Mocap-Driven Simulation                          │
│    for t in timesteps:                              │
│        data.mocap_pos = trajectory[t]               │
│        mujoco.mj_step(model, data)                  │
│                                                      │
│ 4. Inverse Dynamics                                 │
│    mujoco.mj_inverse(model, data)                   │
│    → Compute required joint torques                 │
│                                                      │
│ 5. Physics Data Extraction                          │
│    → Joint torques (26 × T)                         │
│    → Joint velocities                               │
│    → Contact forces (feet-ground)                   │
│    → Club head velocity                             │
│    → Center of pressure                             │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Stage 3: Biomechanics Analysis (10s)                │
├─────────────────────────────────────────────────────┤
│ 1. Kinetic Chain Analysis                           │
│    → Energy transfer efficiency:                    │
│      Legs → Hips → Torso → Shoulders → Arms → Club │
│    → Identify bottlenecks                           │
│                                                      │
│ 2. X-Factor Computation                             │
│    → Shoulder-hip separation angle                  │
│    → Optimal range: 45-55°                          │
│                                                      │
│ 3. Ground Reaction Force Analysis                   │
│    → Left/right foot forces                         │
│    → Weight shift timing                            │
│    → Vertical impulse                               │
│                                                      │
│ 4. Club Metrics                                     │
│    → Club head speed at impact                      │
│    → Attack angle                                   │
│    → Face angle                                     │
│    → Swing path                                     │
│                                                      │
│ 5. Balance & Stability                              │
│    → COP trajectory                                 │
│    → Sway/drift analysis                            │
│    → Stability score                                │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Stage 4: Optimization & Suggestions (15s)           │
├─────────────────────────────────────────────────────┤
│ 1. Load RL Policy (Optional)                        │
│    → Pre-trained PPO model                          │
│    → Generate optimal trajectory                    │
│                                                      │
│ 2. Comparative Analysis                             │
│    → User vs Pro database                           │
│    → User vs RL optimal                             │
│                                                      │
│ 3. Generate Suggestions                             │
│    if energy_efficiency['hips'] < 0.7:              │
│        suggest("Increase hip rotation by 15°")      │
│    if x_factor < 40:                                │
│        suggest("Create more shoulder-hip separation")│
│    if balance_score < 70:                           │
│        suggest("Improve weight transfer timing")    │
│                                                      │
│ 4. Render Visualizations                            │
│    → Annotated video (skeleton overlay)             │
│    → MuJoCo simulation video                        │
│    → Charts: speed curves, torque heatmaps          │
│    → Comparison animations                          │
└─────────────────────────────────────────────────────┘
    ↓
Return Complete Analysis Report
```

### 2.2 Module Architecture

#### AI Vision Module

```python
class PoseEstimator:
    """3D pose estimation from monocular video"""

    def __init__(self):
        self.detector_2d = MediaPipePose()
        self.lifter_3d = VideoPose3D()  # trained model

    def process(self, video_path):
        # Extract frames
        frames = load_video(video_path)

        # 2D detection
        poses_2d = []
        for frame in frames:
            keypoints_2d = self.detector_2d.detect(frame)
            poses_2d.append(keypoints_2d)

        # 3D lifting
        poses_3d = self.lifter_3d.predict(poses_2d)

        return {
            'joints_3d': poses_3d,  # (T, 17, 3)
            'confidence': confidence_scores
        }


class ClubTracker:
    """Golf club detection and tracking"""

    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.tracker = ByteTrack()

    def track(self, video_path):
        # Detect club in each frame
        detections = self.model(video_path, classes=['golf_club'])

        # Track across frames
        trajectories = self.tracker.update(detections)

        return {
            'club_positions': trajectories,
            'club_angles': compute_angles(trajectories)
        }
```

#### MuJoCo Simulation Module

```python
class GolfSwingSimulator:
    """MuJoCo-based physics simulation"""

    def __init__(self, model_path='assets/mjcf/humanoid_golf.xml'):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    def replay_trajectory(self, joint_trajectory):
        """Mocap-driven simulation"""
        physics_data = []

        for t, qpos in enumerate(joint_trajectory):
            # Set mocap targets
            self.data.mocap_pos[:17] = qpos

            # Forward kinematics + inverse dynamics
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_inverse(self.model, self.data)

            # Record physics state
            physics_data.append({
                'time': t * self.model.opt.timestep,
                'qpos': self.data.qpos.copy(),
                'qvel': self.data.qvel.copy(),
                'torques': self.data.qfrc_inverse.copy(),
                'contacts': extract_contact_forces(self.data),
                'club_speed': compute_club_head_speed(self.data)
            })

        return physics_data

    def analyze_kinetic_chain(self, physics_data):
        """Compute energy transfer efficiency"""
        segments = ['legs', 'pelvis', 'torso', 'shoulders', 'arms', 'club']
        energy = {}

        for seg in segments:
            KE = compute_kinetic_energy(physics_data, seg)
            RE = compute_rotational_energy(physics_data, seg)
            energy[seg] = KE + RE

        # Energy transfer ratios
        efficiency = {}
        for i in range(len(segments) - 1):
            ratio = energy[segments[i+1]] / energy[segments[i]]
            efficiency[f"{segments[i]}_to_{segments[i+1]}"] = ratio

        return efficiency


class PhysicsAnalyzer:
    """Biomechanics metrics computation"""

    def compute_x_factor(self, data):
        shoulder_angle = get_rotation(data, 'torso')
        hip_angle = get_rotation(data, 'pelvis')
        return abs(shoulder_angle - hip_angle)

    def compute_grf(self, data):
        left_force = data.contact[0].force
        right_force = data.contact[1].force
        return {
            'left': left_force,
            'right': right_force,
            'total': left_force + right_force,
            'ratio': left_force / (left_force + right_force)
        }

    def compute_club_metrics(self, data, impact_frame):
        club_head_vel = get_site_velocity(data, 'club_head')
        speed = np.linalg.norm(club_head_vel)

        # Convert to mph
        speed_mph = speed * 2.23694

        return {
            'speed_mph': speed_mph,
            'attack_angle': compute_attack_angle(data),
            'face_angle': compute_face_angle(data)
        }
```

#### RL Optimization Module (Advanced)

```python
class GolfSwingEnv(gym.Env):
    """Reinforcement learning environment"""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path('humanoid_golf.xml')
        self.data = mujoco.MjData(self.model)

        # Observation: joint angles, velocities, club position
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (78,))

        # Action: joint torques
        self.action_space = gym.spaces.Box(-1, 1, (26,))

    def step(self, action):
        # Apply torques
        self.data.ctrl[:] = action * 100

        # Simulate
        mujoco.mj_step(self.model, self.data)

        # Compute reward
        club_speed = compute_club_head_speed(self.data)
        balance = compute_balance_score(self.data)
        energy_eff = compute_energy_efficiency(self.data)

        reward = (
            0.5 * normalize(club_speed, 0, 130) +  # Speed: 0-130 mph
            0.3 * balance +                         # Balance: 0-1
            0.2 * energy_eff                        # Efficiency: 0-1
        )

        return self.get_obs(), reward, done, {}


def train_virtual_coach():
    """Train RL policy for optimal swing"""
    env = GolfSwingEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save("assets/rl_models/golf_coach_ppo")
    return model
```

### 2.3 Data Models

```python
# Pydantic models for API

class PoseData(BaseModel):
    """3D pose estimation output"""
    joints_3d: List[List[float]]  # (T, 17, 3)
    confidence: List[float]
    swing_phases: Dict[str, Tuple[int, int]]

class PhysicsMetrics(BaseModel):
    """MuJoCo analysis output"""
    club_head_speed_mph: float
    peak_torques: Dict[str, float]
    energy_efficiency: Dict[str, float]
    x_factor: float
    grf_profile: Dict[str, List[float]]
    balance_score: float

class SwingSuggestion(BaseModel):
    """Optimization suggestion"""
    category: str  # "hip_rotation", "weight_transfer", etc.
    severity: str  # "minor", "moderate", "major"
    message: str
    improvement_potential: float  # Estimated gain in mph

class AnalysisResult(BaseModel):
    """Complete analysis output"""
    video_id: str
    analysis_id: str
    pose_data: PoseData
    physics_metrics: PhysicsMetrics
    suggestions: List[SwingSuggestion]
    visualization_urls: Dict[str, str]
    processing_time: float
```

### 2.4 API Endpoints

```
# Phase 2 API extensions

POST /api/video/analyze/{video_id}
    Request: { "video_id": "uuid" }
    Response: { "analysis_id": "uuid", "status": "queued" }

GET /api/analysis/status/{analysis_id}
    Response: {
        "analysis_id": "uuid",
        "status": "processing" | "completed" | "failed",
        "progress": 0-100,
        "current_stage": "pose_estimation" | "simulation" | "analysis" | "rendering"
    }

GET /api/analysis/result/{analysis_id}
    Response: {
        "analysis_id": "uuid",
        "physics_metrics": { ... },
        "suggestions": [ ... ],
        "visualization_urls": {
            "annotated_video": "url",
            "simulation_video": "url",
            "charts": "url"
        }
    }

GET /api/analysis/visualization/{analysis_id}/{resource}
    resource: "annotated_video" | "simulation_video" | "speed_chart" | "torque_heatmap"
    Response: File download

WebSocket /ws/analysis/{analysis_id}
    Real-time progress updates
```

### 2.5 Async Processing Architecture

```
FastAPI Endpoint
    ↓
Celery Task Queue (Redis)
    ↓
Worker Pool (GPU-enabled)
    ↓
[Task 1] Pose Estimation
[Task 2] MuJoCo Simulation
[Task 3] Visualization Rendering
    ↓
Results stored in Redis
    ↓
WebSocket notification to client
```

## Performance Considerations

### Bottlenecks & Solutions

| Bottleneck | Solution |
|-----------|----------|
| Pose estimation (GPU) | Batch processing, model quantization |
| MuJoCo simulation (CPU) | Parallel workers, C++ optimization |
| Video rendering | FFmpeg GPU encoding, pre-computed templates |
| Large model files | Lazy loading, cloud storage |

### Target Performance Metrics

| Metric | Target |
|--------|--------|
| Total processing time | < 90 seconds |
| Pose estimation | < 30 seconds |
| MuJoCo simulation | < 20 seconds |
| Analysis + rendering | < 30 seconds |
| Concurrent users | 10+ (with async queue) |

## Deployment Architecture

```
┌─────────────────────┐
│   Load Balancer     │
│    (Nginx)          │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐   ┌───▼────┐
│FastAPI │   │FastAPI │  (Multiple instances)
│Worker 1│   │Worker 2│
└───┬────┘   └───┬────┘
    │             │
    └──────┬──────┘
           │
    ┌──────▼──────────┐
    │  Celery Broker  │
    │    (Redis)      │
    └──────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐   ┌───▼────┐
│Celery  │   │Celery  │  (GPU workers)
│Worker 1│   │Worker 2│
└───┬────┘   └───┬────┘
    │             │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Storage    │
    │  (S3/OSS)   │
    └─────────────┘
```

## Security & Privacy

- Video files encrypted at rest
- UUID-based access control
- Rate limiting on API endpoints
- User authentication (Phase 2)
- HTTPS enforced in production
- Video auto-deletion after 30 days

## Monitoring & Logging

```python
# Structured logging
logger.info("analysis_started", extra={
    "video_id": video_id,
    "analysis_id": analysis_id,
    "file_size_mb": file_size
})

# Metrics collection
metrics.timing("pose_estimation.duration", duration)
metrics.increment("analysis.completed")
metrics.gauge("queue.depth", queue_size)

# Error tracking
sentry.capture_exception(error, context={
    "video_id": video_id,
    "stage": "mujoco_simulation"
})
```

## Future Enhancements

- Multi-camera support (stereo depth)
- Real-time analysis (edge deployment)
- Comparison with pro database
- Progressive training plans
- AR overlay in WeChat mini program
- Multiplayer challenges/competitions

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MuJoCo Python Bindings](https://mujoco.readthedocs.io/)
- [Celery Documentation](https://docs.celeryq.dev/)
- [VideoPose3D Paper](https://arxiv.org/abs/1811.11742)
