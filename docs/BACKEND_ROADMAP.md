# 后端研发路线图

## 概述

本文档规划后端研发路线，目标是通过本项目获得以下实践经验：

1. **深度学习** — MLP、CNN、RNN、Transformer
2. **强化学习** — Q-Learning、PPO
3. **机器人仿真** — MuJoCo

三个阶段层层递进：MuJoCo 提供仿真基础，DL 模型消费仿真数据，RL 在仿真环境中训练智能体。

---

## 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| MediaPipe 姿态估计 | 可用 | 从视频提取 17 个 COCO 关键点；调用现成 API，非自研 DL |
| MuJoCo 仿真器 | 仅脚手架 | 模型可加载，但 IK 映射为占位（仅 1 个关节），`ctrl[:] = 0`，回退到 dummy 数据 |
| 深度学习 | 无 | `nn.Module` 继承仅为形式，无实际神经网络层或训练逻辑 |
| 强化学习 | 无 | `GolfSwingEnv` 仅存在于 ARCHITECTURE.md 伪代码中 |

---

## 阶段一：MuJoCo 仿真（基础）

**目标**：让 humanoid 在 MuJoCo 中真正复现挥杆动作，产出有意义的物理数据。

### 1.1 逆运动学映射

**文件**：`backend/app/models/mujoco_simulator.py` → `_map_pose_to_joint_angles()`

当前状态：仅映射了一个关节，使用粗略启发式。需要：

- 将所有 17 个 COCO 关键点映射到 humanoid 的 26 DOF 关节角度
- 使用 `atan2` 从肢体向量计算关节角度（如从上臂/前臂向量计算肘关节屈伸角）
- 处理 MediaPipe（相机坐标系）与 MuJoCo（世界坐标系）之间的坐标对齐

### 1.2 PD 控制器

**文件**：`backend/app/models/mujoco_simulator.py` → `_simulate_trajectory()`

当前状态：`self.data.ctrl[:] = 0`（零力矩）。需要：

- 实现比例-微分控制：
  ```
  ctrl = Kp * (q_target - q_actual) + Kd * (dq_target - dq_actual)
  ```
- 按关节调节 Kp/Kd 增益（大关节需要更高增益）
- 确保跟踪稳定，无振荡或发散

### 1.3 物理数据提取

控制器正常工作后，提取有意义的：

- 关节力矩（逆动力学）
- 地面反作用力（足-地接触）
- 质心轨迹
- 击球帧的杆头速度

### 交付物

上传视频 → MuJoCo humanoid 复现挥杆 → 物理指标为真实数据而非 dummy。

---

## 阶段二：深度学习（四种架构）

**目标**：在分析管线中应用 MLP、CNN、RNN 和 Transformer 解决实际问题。

### 2.1 MLP — 学习逆运动学

**用途**：用神经网络替代手写 IK 映射。

- **输入**：17 × 3 = 51 维关键点坐标（来自 MediaPipe）
- **输出**：26 维关节角度（MuJoCo humanoid）
- **训练数据**：通过 MuJoCo 正运动学自监督生成
  - 采样随机关节角度 → 运行 FK → 得到关键点位置 → (关键点, 角度) 数据对
- **架构**：3-4 层隐藏层的 MLP，ReLU 激活，BatchNorm
- **损失函数**：关节角度 MSE + 可选 FK 一致性损失
- **文件**：新建 `backend/app/models/learned_ik.py`

### 2.2 CNN — 挥杆阶段检测

**用途**：将视频帧分类为 5 个挥杆阶段。

- **类别**：准备 (Address)、上杆 (Backswing)、下杆 (Downswing)、击球 (Impact)、送杆 (Follow-through)
- **输入**：单帧视频图像（缩放至 224×224）
- **架构**：轻量 CNN（自定义小网络或微调 MobileNet）
- **训练数据**：手动标注约 500 帧（或使用姿态数据启发式预标注）
- **文件**：新建 `backend/app/models/swing_phase_cnn.py`

### 2.3 RNN (LSTM) — 挥杆质量评估

**用途**：基于关节角度时间序列评估挥杆质量。

- **输入**：逐帧关节角度序列，形状 (T, 26)
- **输出**：连续质量评分（速度、平衡、效率）
- **架构**：2 层 LSTM + 全连接输出头
- **训练数据**：MuJoCo 物理指标作为标签，关节角度序列作为输入
- **文件**：新建 `backend/app/models/swing_scorer_rnn.py`

### 2.4 Transformer — 动作预测与异常检测

**用途**：预测未来姿态，检测异常挥杆模式。

- **输入**：过去 N 帧关节数据窗口
- **输出**：未来 M 帧预测；异常分数
- **架构**：小型 encoder-only Transformer（4 层，4 头）
- **训练方法**：自监督下一帧预测；重建误差高 = 异常
- **文件**：新建 `backend/app/models/motion_transformer.py`

### 交付物

分析结果包含：自动识别的挥杆阶段、质量评分、姿势异常警告 — 均由训练好的神经网络驱动。

---

## 阶段三：强化学习

**目标**：在 MuJoCo 中训练 RL 智能体学习最优挥杆，与用户挥杆对比。

### 3.1 Gym 环境

**文件**：新建 `backend/app/models/golf_swing_env.py`

- **状态空间**：关节角度 (26) + 关节速度 (26) + 杆头位置 (3) + 阶段指示 ≈ 56 维
- **动作空间**：连续关节力矩（26 维，归一化至 [-1, 1]）
- **奖励函数**：
  ```
  R = w1 * club_head_speed       # 最大化击球速度
    + w2 * balance_score          # 保持稳定
    + w3 * energy_efficiency      # 高效动力链
    - w4 * joint_limit_penalty    # 不超关节限位
    - w5 * fall_penalty           # 不摔倒
  ```
- **回合**：从准备姿势开始，到送杆完成或摔倒结束
- **重置**：回到中立站姿，持杆

### 3.2 PPO 训练（主要算法）

- 使用 Stable-Baselines3 的 `PPO` + `MlpPolicy`
- 连续动作空间 — MuJoCo 控制任务的标配算法
- 需调节的超参数：学习率、裁剪范围、GAE lambda、并行环境数
- 目标：约 100 万步初始策略
- **文件**：新建 `backend/app/models/rl_trainer.py`

### 3.3 Q-Learning（对比实验）

- 对问题进行离散化以作对比：
  - 简化动作空间：如每个主要关节组（髋、肩、腕）3 个力度等级 → 可管理的离散空间
  - 或对简化的状态/动作表示应用 DQN
- 目的：理解基于价值（Q-Learning）和基于策略（PPO）方法的差异
- **文件**：新建 `backend/app/models/rl_qlearning.py`

### 3.4 应用：虚拟教练

- 将用户挥杆轨迹与 RL 学到的最优轨迹对比
- 生成具体改进建议：
  - "下杆起始时，您的髋部旋转比最优值少 15°"
  - "重心向前脚转移的时机晚了 0.1 秒"
- 可视化用户挥杆与最优挥杆的并排对比

### 交付物

训练好的 RL 智能体能在 MuJoCo 中完成挥杆；通过与最优策略对比，为用户提供个性化教练建议。

---

## 实施优先级

```
阶段 1.2  PD 控制器           ← 从这里开始（立竿见影，解锁后续所有工作）
阶段 1.1  IK 映射             ← 完善关键点到关节角度的映射
阶段 1.3  物理数据提取         ← 用真实数据验证
阶段 2.1  MLP（学习 IK）      ← 第一个 DL 模型，替代手写 IK
阶段 2.2  CNN（阶段检测）     ← 不依赖 MuJoCo，可并行开发
阶段 2.3  RNN（质量评估）     ← 需要阶段 1 的数据
阶段 2.4  Transformer（动作） ← 最高级的 DL 组件
阶段 3.1  Gym 环境            ← 基于已运行的 MuJoCo 仿真
阶段 3.2  PPO 训练            ← 主要 RL 算法
阶段 3.3  Q-Learning          ← 对比实验
阶段 3.4  虚拟教练            ← 集成与最终产品
```

---

## 关键依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| mujoco | >= 3.0.0 | 物理仿真 |
| torch | >= 2.1 | 深度学习框架 |
| stable-baselines3 | >= 2.0 | PPO / DQN 实现 |
| gymnasium | >= 0.29 | RL 环境接口 |
| mediapipe | >= 0.10 | 姿态估计（已有） |
| scipy | >= 1.10 | 信号处理 |

---

## 参考资料

- MuJoCo 文档: https://mujoco.readthedocs.io/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- DeepMimic（基于 RL 的动作模仿）: https://arxiv.org/abs/1804.02717
- VideoPose3D（2D 到 3D 姿态提升）: https://arxiv.org/abs/1811.11742
