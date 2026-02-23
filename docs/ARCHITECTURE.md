# 技术架构

## 系统概述

本文档描述 AI 高尔夫挥杆分析系统的技术架构，涵盖 Phase 1（已实现）和 Phase 2（规划中）。

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       微信小程序                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  录制页面 │→ │  上传页面 │→ │  处理中   │→ │  结果页面 │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│         │              │              ↓              ↑           │
└─────────┼──────────────┼──────────────┼──────────────┼──────────┘
          │              │              │              │
          │   HTTP/HTTPS │              │ WebSocket    │
          │              │              │ (Phase 2)    │
          ↓              ↓              ↓              │
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI 后端                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API 层 (routes/)                       │  │
│  │   /video/upload  │  /video/status  │  /analysis/result   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  服务层 (services/)                       │  │
│  │  VideoService  │  GolfAnalysisService  │  VisualizationSvc│  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  模型层 (models/)                         │  │
│  │  ┌───────────┐  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │  Phase 1  │  │   Phase 2A   │  │    Phase 2B      │  │  │
│  │  │  Dummy    │  │  AI 视觉     │  │  RL 优化器       │  │  │
│  │  │  处理器   │  │  + MuJoCo    │  │                  │  │  │
│  │  └───────────┘  └──────────────┘  └──────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              存储与处理引擎                                │  │
│  │  文件系统  │  Celery 队列  │  Redis 缓存  │  GPU         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1：视频处理管线（已实现）

### 1.1 前端流程

```
用户操作 → 相机 API → 视频录制 → 临时文件
                                       ↓
                               wx.uploadFile()
                                       ↓
                               后端 /upload
                                       ↓
                         轮询 /status/{video_id}
                                       ↓
                         下载 /result/{video_id}
```

### 1.2 后端处理

```python
# 视频上传 → 处理 → 存储
POST /api/video/upload
    ↓
VideoService.save_upload(video_id, content)
    → 保存至: data/raw_videos/{video_id}.mp4
    ↓
VideoService.process_video(video_id)
    ↓
DummyVideoProcessor.forward(input_path, output_path)
    → OpenCV: 添加水印
    → FFmpeg: 重新编码为 H.264+AAC
    → 保存至: data/processed_videos/{video_id}.mp4
    ↓
生成元数据 JSON
    → 保存至: data/metadata/{video_id}.json
    ↓
返回: {video_id, status: "done"}
```

### 1.3 数据流

```
┌─────────────┐
│  原始视频    │ → 上传 → data/raw_videos/{id}.mp4
└─────────────┘
       ↓
┌─────────────┐
│  处理阶段    │ → OpenCV + FFmpeg
└─────────────┘
       ↓
┌─────────────────┐
│  处理后视频      │ → data/processed_videos/{id}.mp4
└─────────────────┘
       ↓
┌─────────────┐
│  元数据      │ → data/metadata/{id}.json
└─────────────┘
```

### 1.4 关键技术

| 组件 | 技术 | 用途 |
|------|------|------|
| Web 框架 | FastAPI | REST API 端点 |
| 视频 I/O | OpenCV | 帧读写 |
| 编解码 | FFmpeg | H.264+AAC 编码 |
| 深度学习框架 | PyTorch | 模型基础（Phase 1 为 Dummy） |
| 序列化 | Pydantic | 数据验证 |

---

## Phase 2：AI + 物理分析管线（规划中）

### 2.1 分析管线

```
视频上传
    ↓
┌─────────────────────────────────────────────────────┐
│ 阶段 1：AI 视频分析（约30秒）                         │
├─────────────────────────────────────────────────────┤
│ 1. 2D 姿态检测（MediaPipe）                           │
│    → 每帧检测 17 个关键点                              │
│                                                      │
│ 2. 3D 姿态提升（VideoPose3D）                         │
│    → 2D → 3D 轨迹                                    │
│    → 输出: (T, 17, 3) 关节坐标                        │
│                                                      │
│ 3. 物体检测（YOLOv8）                                 │
│    → 追踪球杆轨迹                                     │
│    → 追踪高尔夫球（如可见）                             │
│                                                      │
│ 4. 时序分割                                           │
│    → 识别挥杆阶段:                                     │
│      - 准备 (Address): [0, t1]                        │
│      - 上杆 (Backswing): [t1, t2]                     │
│      - 下杆 (Downswing): [t2, t3]                     │
│      - 击球 (Impact): [t3, t4]                        │
│      - 送杆 (Follow-through): [t4, T]                 │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 阶段 2：MuJoCo 物理仿真（约20秒）                     │
├─────────────────────────────────────────────────────┤
│ 1. 加载模型                                          │
│    → humanoid_golf.xml（26 DOF + 球杆）               │
│                                                      │
│ 2. 轨迹平滑                                          │
│    → Savitzky-Golay 滤波器                            │
│    → 消除 AI 预测抖动                                  │
│                                                      │
│ 3. Mocap 驱动仿真                                     │
│    for t in timesteps:                               │
│        data.mocap_pos = trajectory[t]                │
│        mujoco.mj_step(model, data)                   │
│                                                      │
│ 4. 逆动力学                                          │
│    mujoco.mj_inverse(model, data)                    │
│    → 计算所需关节力矩                                  │
│                                                      │
│ 5. 物理数据提取                                       │
│    → 关节力矩 (26 × T)                               │
│    → 关节速度                                         │
│    → 接触力（足-地）                                   │
│    → 杆头速度                                         │
│    → 压力中心                                         │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 阶段 3：生物力学分析（约10秒）                         │
├─────────────────────────────────────────────────────┤
│ 1. 动力链分析                                         │
│    → 能量传递效率:                                     │
│      腿 → 髋 → 躯干 → 肩 → 手臂 → 球杆               │
│    → 识别瓶颈                                         │
│                                                      │
│ 2. X-Factor 计算                                      │
│    → 肩髋分离角                                       │
│    → 最优范围: 45-55°                                 │
│                                                      │
│ 3. 地面反作用力分析                                    │
│    → 左/右脚力                                        │
│    → 重心转移时机                                      │
│    → 垂直冲量                                         │
│                                                      │
│ 4. 球杆指标                                           │
│    → 击球瞬间杆头速度                                  │
│    → 攻角                                             │
│    → 杆面角度                                         │
│    → 挥杆路径                                         │
│                                                      │
│ 5. 平衡与稳定性                                       │
│    → 压力中心轨迹                                      │
│    → 晃动/漂移分析                                     │
│    → 稳定性评分                                       │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ 阶段 4：优化与建议（约15秒）                           │
├─────────────────────────────────────────────────────┤
│ 1. 加载 RL 策略（可选）                                │
│    → 预训练 PPO 模型                                   │
│    → 生成最优轨迹                                      │
│                                                      │
│ 2. 对比分析                                           │
│    → 用户 vs 职业选手数据库                             │
│    → 用户 vs RL 最优策略                               │
│                                                      │
│ 3. 生成建议                                           │
│    if energy_efficiency['hips'] < 0.7:               │
│        suggest("增加髋部旋转 15°")                     │
│    if x_factor < 40:                                 │
│        suggest("增大肩髋分离角")                       │
│    if balance_score < 70:                            │
│        suggest("改善重心转移时机")                      │
│                                                      │
│ 4. 渲染可视化                                         │
│    → 标注视频（骨骼叠加）                               │
│    → MuJoCo 仿真视频                                  │
│    → 图表: 速度曲线、力矩热力图                         │
│    → 对比动画                                         │
└─────────────────────────────────────────────────────┘
    ↓
返回完整分析报告
```

### 2.2 模块架构

#### AI 视觉模块

```python
class PoseEstimator:
    """单目视频 3D 姿态估计"""

    def __init__(self):
        self.detector_2d = MediaPipePose()
        self.lifter_3d = VideoPose3D()  # 预训练模型

    def process(self, video_path):
        # 提取视频帧
        frames = load_video(video_path)

        # 2D 检测
        poses_2d = []
        for frame in frames:
            keypoints_2d = self.detector_2d.detect(frame)
            poses_2d.append(keypoints_2d)

        # 3D 提升
        poses_3d = self.lifter_3d.predict(poses_2d)

        return {
            'joints_3d': poses_3d,  # (T, 17, 3)
            'confidence': confidence_scores
        }


class ClubTracker:
    """球杆检测与追踪"""

    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.tracker = ByteTrack()

    def track(self, video_path):
        # 逐帧检测球杆
        detections = self.model(video_path, classes=['golf_club'])

        # 跨帧追踪
        trajectories = self.tracker.update(detections)

        return {
            'club_positions': trajectories,
            'club_angles': compute_angles(trajectories)
        }
```

#### MuJoCo 仿真模块

```python
class GolfSwingSimulator:
    """基于 MuJoCo 的物理仿真"""

    def __init__(self, model_path='assets/mjcf/humanoid_golf.xml'):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    def replay_trajectory(self, joint_trajectory):
        """Mocap 驱动仿真"""
        physics_data = []

        for t, qpos in enumerate(joint_trajectory):
            # 设置 mocap 目标
            self.data.mocap_pos[:17] = qpos

            # 正向运动学 + 逆动力学
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_inverse(self.model, self.data)

            # 记录物理状态
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
        """计算能量传递效率"""
        segments = ['legs', 'pelvis', 'torso', 'shoulders', 'arms', 'club']
        energy = {}

        for seg in segments:
            KE = compute_kinetic_energy(physics_data, seg)
            RE = compute_rotational_energy(physics_data, seg)
            energy[seg] = KE + RE

        # 能量传递比
        efficiency = {}
        for i in range(len(segments) - 1):
            ratio = energy[segments[i+1]] / energy[segments[i]]
            efficiency[f"{segments[i]}_to_{segments[i+1]}"] = ratio

        return efficiency


class PhysicsAnalyzer:
    """生物力学指标计算"""

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

        # 转换为英里/小时
        speed_mph = speed * 2.23694

        return {
            'speed_mph': speed_mph,
            'attack_angle': compute_attack_angle(data),
            'face_angle': compute_face_angle(data)
        }
```

#### RL 优化模块（高级功能）

```python
class GolfSwingEnv(gym.Env):
    """强化学习环境"""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path('humanoid_golf.xml')
        self.data = mujoco.MjData(self.model)

        # 观测空间: 关节角度、速度、球杆位置
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (78,))

        # 动作空间: 关节力矩
        self.action_space = gym.spaces.Box(-1, 1, (26,))

    def step(self, action):
        # 施加力矩
        self.data.ctrl[:] = action * 100

        # 仿真一步
        mujoco.mj_step(self.model, self.data)

        # 计算奖励
        club_speed = compute_club_head_speed(self.data)
        balance = compute_balance_score(self.data)
        energy_eff = compute_energy_efficiency(self.data)

        reward = (
            0.5 * normalize(club_speed, 0, 130) +  # 速度: 0-130 mph
            0.3 * balance +                         # 平衡: 0-1
            0.2 * energy_eff                        # 效率: 0-1
        )

        return self.get_obs(), reward, done, {}


def train_virtual_coach():
    """训练最优挥杆 RL 策略"""
    env = GolfSwingEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save("assets/rl_models/golf_coach_ppo")
    return model
```

### 2.3 数据模型

```python
# Pydantic API 数据模型

class PoseData(BaseModel):
    """3D 姿态估计输出"""
    joints_3d: List[List[float]]  # (T, 17, 3)
    confidence: List[float]
    swing_phases: Dict[str, Tuple[int, int]]

class PhysicsMetrics(BaseModel):
    """MuJoCo 分析输出"""
    club_head_speed_mph: float
    peak_torques: Dict[str, float]
    energy_efficiency: Dict[str, float]
    x_factor: float
    grf_profile: Dict[str, List[float]]
    balance_score: float

class SwingSuggestion(BaseModel):
    """优化建议"""
    category: str  # "hip_rotation", "weight_transfer" 等
    severity: str  # "minor", "moderate", "major"
    message: str
    improvement_potential: float  # 预估速度提升（mph）

class AnalysisResult(BaseModel):
    """完整分析结果"""
    video_id: str
    analysis_id: str
    pose_data: PoseData
    physics_metrics: PhysicsMetrics
    suggestions: List[SwingSuggestion]
    visualization_urls: Dict[str, str]
    processing_time: float
```

### 2.4 API 端点

```
# Phase 2 API 扩展

POST /api/video/analyze/{video_id}
    请求: { "video_id": "uuid" }
    响应: { "analysis_id": "uuid", "status": "queued" }

GET /api/analysis/status/{analysis_id}
    响应: {
        "analysis_id": "uuid",
        "status": "processing" | "completed" | "failed",
        "progress": 0-100,
        "current_stage": "pose_estimation" | "simulation" | "analysis" | "rendering"
    }

GET /api/analysis/result/{analysis_id}
    响应: {
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
    响应: 文件下载

WebSocket /ws/analysis/{analysis_id}
    实时进度推送
```

### 2.5 异步处理架构

```
FastAPI 端点
    ↓
Celery 任务队列 (Redis)
    ↓
Worker 池（GPU）
    ↓
[任务 1] 姿态估计
[任务 2] MuJoCo 仿真
[任务 3] 可视化渲染
    ↓
结果存储至 Redis
    ↓
WebSocket 通知客户端
```

## 性能考量

### 瓶颈与解决方案

| 瓶颈 | 解决方案 |
|------|---------|
| 姿态估计（GPU） | 批处理、模型量化 |
| MuJoCo 仿真（CPU） | 并行 Worker、C++ 优化 |
| 视频渲染 | FFmpeg GPU 编码、预计算模板 |
| 大模型文件 | 懒加载、云存储 |

### 目标性能指标

| 指标 | 目标 |
|------|------|
| 总处理时间 | < 90 秒 |
| 姿态估计 | < 30 秒 |
| MuJoCo 仿真 | < 20 秒 |
| 分析 + 渲染 | < 30 秒 |
| 并发用户 | 10+（使用异步队列） |

## 部署架构

```
┌─────────────────────┐
│   负载均衡            │
│    (Nginx)           │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐   ┌───▼────┐
│FastAPI │   │FastAPI │  （多实例）
│Worker 1│   │Worker 2│
└───┬────┘   └───┬────┘
    │             │
    └──────┬──────┘
           │
    ┌──────▼──────────┐
    │  Celery Broker   │
    │    (Redis)       │
    └──────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐   ┌───▼────┐
│Celery  │   │Celery  │  （GPU Worker）
│Worker 1│   │Worker 2│
└───┬────┘   └───┬────┘
    │             │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  存储        │
    │  (S3/OSS)   │
    └─────────────┘
```

## 安全与隐私

- 视频文件静态加密
- 基于 UUID 的访问控制
- API 端点限流
- 用户认证（Phase 2）
- 生产环境强制 HTTPS
- 视频 30 天自动删除

## 监控与日志

```python
# 结构化日志
logger.info("analysis_started", extra={
    "video_id": video_id,
    "analysis_id": analysis_id,
    "file_size_mb": file_size
})

# 指标采集
metrics.timing("pose_estimation.duration", duration)
metrics.increment("analysis.completed")
metrics.gauge("queue.depth", queue_size)

# 错误追踪
sentry.capture_exception(error, context={
    "video_id": video_id,
    "stage": "mujoco_simulation"
})
```

## 未来扩展

- 多摄像头支持（双目深度）
- 实时分析（边缘部署）
- 职业选手数据库对比
- 渐进式训练计划
- 微信小程序 AR 叠加
- 多人挑战/竞赛

## 参考资料

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [MuJoCo Python 绑定](https://mujoco.readthedocs.io/)
- [Celery 文档](https://docs.celeryq.dev/)
- [VideoPose3D 论文](https://arxiv.org/abs/1811.11742)
