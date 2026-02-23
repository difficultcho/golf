# MuJoCo 模型设计文档

## 概述

本文档描述用于高尔夫挥杆仿真的 MuJoCo MJCF 模型设计。该模型将 26 自由度人体与球杆附件相结合，用于真实的物理仿真。

## 模型规格

### 1. 人体结构

#### 身体段（9 个主体）

```
root（骨盆）
├── torso（躯干）
│   ├── left_shoulder（左肩）
│   │   ├── left_upper_arm（左上臂）
│   │   │   ├── left_lower_arm（左前臂）
│   │   │   │   └── left_hand（左手）
│   │   │   │       └── golf_club（球杆附件）
│   ├── right_shoulder（右肩）
│   │   └── ...
│   └── head（头部）
├── left_hip（左髋）
│   ├── left_thigh（左大腿）
│   │   ├── left_shin（左小腿）
│   │   │   └── left_foot（左脚）
└── right_hip（右髋）
    └── ...
```

#### 自由度（26 DOF）

| 身体部位 | 关节类型 | DOF | 运动范围 |
|---------|---------|-----|---------|
| **脊柱/躯干** | | | |
| 腰椎屈伸 | 铰链 | 1 | -30° ~ 60° |
| 腰椎旋转 | 铰链 | 1 | -45° ~ 45° |
| 胸椎旋转 | 铰链 | 1 | -35° ~ 35° |
| **肩部**（×2） | | | |
| 肩部屈伸 | 铰链 | 1 | -60° ~ 180° |
| 肩部外展 | 铰链 | 1 | 0° ~ 180° |
| 肩部旋转 | 铰链 | 1 | -90° ~ 90° |
| **手臂**（×2） | | | |
| 肘部屈伸 | 铰链 | 1 | 0° ~ 145° |
| 前臂旋转 | 铰链 | 1 | -90° ~ 90° |
| **手腕**（×2） | | | |
| 手腕屈伸 | 铰链 | 1 | -70° ~ 80° |
| 手腕桡偏 | 铰链 | 1 | -25° ~ 25° |
| **髋部**（×2） | | | |
| 髋部屈伸 | 铰链 | 1 | -30° ~ 120° |
| 髋部外展 | 铰链 | 1 | -45° ~ 45° |
| 髋部旋转 | 铰链 | 1 | -45° ~ 45° |
| **膝部**（×2） | | | |
| 膝部屈伸 | 铰链 | 1 | 0° ~ 135° |
| **踝部**（×2） | | | |
| 踝部屈伸 | 铰链 | 1 | -30° ~ 45° |

**挥杆关键关节**：
- 脊柱旋转（产生力矩）
- 肩部复合体（手臂路径）
- 手腕铰链/释放（释放时机）
- 髋部旋转（力量产生）

### 2. 球杆模型

#### 物理属性

```xml
<body name="golf_club" pos="0.05 0 -0.1">
  <joint type="ball" damping="0.5" armature="0.01"/>

  <!-- 杆身 -->
  <geom name="club_shaft" type="capsule"
        fromto="0 0 0 0 0 -1.0"
        size="0.01"
        mass="0.3"
        rgba="0.3 0.3 0.3 1"/>

  <!-- 杆头（一号木） -->
  <geom name="club_head" type="box"
        pos="0 0.05 -1.0"
        size="0.02 0.05 0.03"
        mass="0.2"
        rgba="0.2 0.2 0.2 1"/>
</body>
```

**标准一号木参数**：
- 总长度：1.15m（45 英寸）
- 杆身质量：60-70g
- 杆头质量：约 200g
- 重心位置：杆面后方约 35mm
- 转动惯量：约 5000 g·cm²

### 3. 接触模型

#### 地面接触

```xml
<geom name="floor" type="plane"
      size="5 5 0.1"
      pos="0 0 0"
      material="floor_mat"/>

<material name="floor_mat"
          friction="1.0 0.005 0.0001"
          condim="3"/>
```

**摩擦参数**：
- μ₁（切向）：1.0（草地/果岭）
- μ₂（扭转）：0.005
- μ₃（滚动）：0.0001

#### 足-地接触

```xml
<geom name="left_foot" type="box"
      size="0.1 0.05 0.02"
      condim="3"
      friction="1.0 0.005 0.0001"/>
```

### 4. 执行器（关节控制器）

#### 力矩控制执行器

```xml
<actuator>
  <!-- 脊柱 -->
  <motor joint="lumbar_rotation" gear="150" ctrllimited="true" ctrlrange="-150 150"/>
  <motor joint="thoracic_rotation" gear="120" ctrllimited="true" ctrlrange="-120 120"/>

  <!-- 肩部 -->
  <motor joint="left_shoulder_flex" gear="100" ctrllimited="true" ctrlrange="-100 100"/>
  <motor joint="left_shoulder_rot" gear="80" ctrllimited="true" ctrlrange="-80 80"/>

  <!-- 髋部 -->
  <motor joint="left_hip_rotation" gear="150" ctrllimited="true" ctrlrange="-150 150"/>

  <!-- 手腕（挥杆关键） -->
  <motor joint="left_wrist_flex" gear="50" ctrllimited="true" ctrlrange="-50 50"/>

  <!-- ...（共 26 个执行器） -->
</actuator>
```

**齿轮比（最大力矩，Nm）**：
- 脊柱/髋部：150 Nm（大肌群）
- 肩部：100 Nm
- 肘部：80 Nm
- 手腕：50 Nm
- 踝部：60 Nm

### 5. Mocap 体（轨迹跟踪）

```xml
<body name="mocap_pelvis" mocap="true">
  <geom type="sphere" size="0.02" rgba="1 0 0 0.3" contype="0" conaffinity="0"/>
</body>

<!-- 17 个 mocap 体，对应姿态估计关键点 -->
<body name="mocap_left_wrist" mocap="true">
  <geom type="sphere" size="0.015" rgba="0 1 0 0.3" contype="0" conaffinity="0"/>
</body>
<!-- ... -->

<!-- 等式约束：跟踪 mocap 位置 -->
<equality>
  <weld body1="pelvis" body2="mocap_pelvis" solref="0.01 1"/>
  <weld body1="left_hand" body2="mocap_left_wrist" solref="0.01 1"/>
  <!-- ... -->
</equality>
```

**Mocap 关键点**（17 个，对应 VideoPose3D 输出）：
- 头部、颈部
- 左/右肩
- 左/右肘
- 左/右腕
- 骨盆
- 左/右髋
- 左/右膝
- 左/右踝

## 仿真模式

### 模式 1：轨迹回放（Mocap 驱动）

**应用场景**：从视频回放用户的挥杆动作

```python
# 使用 AI 姿态估计的结果设置 mocap 位置
for t, poses in enumerate(trajectory):
    data.mocap_pos[:17] = poses  # 17 个关键点 × 3D
    mujoco.mj_step(model, data)

    # 逆动力学：计算所需力矩
    mujoco.mj_inverse(model, data)
    joint_torques = data.qfrc_inverse.copy()
```

**关键计算**：
- 逆动力学 → 关节力矩
- 地面反作用力
- 压力中心
- 身体段间能量传递

### 模式 2：正向动力学（RL 驱动）

**应用场景**：训练最优挥杆策略

```python
# 施加 RL 策略输出的力矩命令
data.ctrl[:] = action  # 26 个执行器命令
mujoco.mj_step(model, data)

# 计算奖励
club_speed = compute_club_head_speed(data)
accuracy = compute_impact_accuracy(data)
balance = compute_balance_score(data)

reward = w1*club_speed + w2*accuracy + w3*balance
```

## 物理分析指标

### 1. 动力链分析

测量能量在身体段间的传递：

```python
def compute_energy_transfer(data, timesteps):
    segments = ['legs', 'pelvis', 'torso', 'shoulders', 'arms', 'wrists', 'club']

    for seg in segments:
        # 动能
        KE[seg] = 0.5 * mass[seg] * velocity[seg]**2

        # 转动能
        RE[seg] = 0.5 * I[seg] * omega[seg]**2

        total_energy[seg] = KE[seg] + RE[seg]

    # 能量传递效率
    efficiency = total_energy['club'] / total_energy['legs']

    # 识别瓶颈
    for i in range(len(segments)-1):
        transfer_ratio = total_energy[segments[i+1]] / total_energy[segments[i]]
        if transfer_ratio < 0.7:
            bottleneck = segments[i]
```

### 2. X-Factor（肩髋分离角）

```python
def compute_x_factor(data):
    shoulder_angle = get_body_angle(data, 'torso', 'frontal_plane')
    hip_angle = get_body_angle(data, 'pelvis', 'frontal_plane')

    x_factor = abs(shoulder_angle - hip_angle)

    # 最优范围：上杆顶点时 45-55 度
    return x_factor
```

### 3. 地面反作用力

```python
def analyze_grf(data):
    left_foot_force = data.contact[left_foot_id].force
    right_foot_force = data.contact[right_foot_id].force

    total_grf = left_foot_force + right_foot_force

    # 重心转移分析
    weight_ratio = left_foot_force / right_foot_force

    # 垂直力曲线
    vertical_impulse = integrate(total_grf[2], dt)
```

### 4. 杆头速度

```python
def compute_club_head_speed(data):
    club_head_pos = data.geom('club_head').xpos
    club_head_vel = numerical_derivative(club_head_pos, dt)

    speed = np.linalg.norm(club_head_vel)

    # 典型范围：
    # 业余选手: 80-95 mph
    # 低差点: 95-105 mph
    # 职业选手: 110-125 mph

    return speed  # 单位 mph
```

## 模型验证

### 物理约束
- 关节限位符合人体解剖学范围
- 身体段质量符合人体测量学数据
- 球杆参数符合标准一号木规格
- 地面摩擦符合草地条件

### 稳定性测试
- 静态站立平衡（无漂移）
- 静态姿势保持（无抖动）
- 轨迹跟踪平滑（无异常力）
- 能量守恒（误差 < 5%）

### 与真实数据对比
- 杆头速度：与实测值误差 ±10%
- 关节力矩峰值：与 EMG 推断值匹配
- 地面反作用力曲线：与力板数据匹配
- X-Factor：处于职业选手范围内

## 实施路线图

### Phase 2A：基础模型
1. 创建 26 DOF 人体 MJCF 文件
2. 添加球杆附件
3. 实现 mocap 跟踪
4. 测试轨迹回放

### Phase 2B：物理分析
1. 实现逆动力学管线
2. 添加地面反作用力和压力中心计算
3. 实现动力链分析
4. 与真实数据验证

### Phase 2C：RL 环境
1. 定义观测/动作空间
2. 实现奖励函数
3. 训练 PPO 策略
4. 部署用于优化

## 参考资料

### MuJoCo 资源
- [MuJoCo 文档](https://mujoco.readthedocs.io/)
- [MJCF XML 参考](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [Humanoid 模型示例](https://github.com/google-deepmind/mujoco/tree/main/model)

### 生物力学文献
- Myers, J., et al. (2008). "The role of the hip in the golf swing"
- Kwon, Y. H., et al. (2012). "The mechanism of kinematic sequence in golf swing"
- Cheetham, P. J., et al. (2001). "Comparison of kinematic sequence parameters"

### 高尔夫物理学
- Nesbit, S. M. (2005). "A three dimensional kinematic and kinetic study of the golf swing"
- Penner, A. R. (2003). "The physics of golf"
