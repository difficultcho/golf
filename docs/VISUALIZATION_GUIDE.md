# MuJoCo 可视化指南

## 概述

本项目提供三种MuJoCo可视化方式，用于开发和测试高尔夫挥杆物理仿真。

## 🎯 快速开始

### 前置条件

```bash
cd backend
source venv/bin/activate
pip install matplotlib imageio imageio-ffmpeg
```

### 方法1：物理数据曲线图（推荐用于分析）

**生成物理指标的可视化图表**

```bash
python visualize_mujoco.py --mode plot
```

**输出：**
- `physics_analysis.png` - 包含4个子图的分析图表：
  - 杆头速度 (mph)
  - 骨盆高度 (m) - 平衡指标
  - 腰部旋转角度 (°) - X-Factor
  - 地面反作用力 (N)

**适用场景：**
- ✅ 分析仿真的物理正确性
- ✅ 调试模型参数
- ✅ 生成技术报告
- ✅ 无需GUI环境（可在服务器上运行）

---

### 方法2：3D交互式查看器（推荐用于调试）

**实时3D可视化，可交互控制**

```bash
python visualize_mujoco.py --mode viewer
```

**交互控制：**
| 操作 | 功能 |
|------|------|
| 鼠标左键拖动 | 旋转视角 |
| 鼠标滚轮 | 缩放 |
| 鼠标右键拖动 | 平移视角 |
| 空格键 | 暂停/恢复仿真 |
| Backspace | 重置到初始姿势 |
| ESC | 关闭查看器 |

**适用场景：**
- ✅ 直观查看模型动作
- ✅ 调试关节运动范围
- ✅ 验证碰撞检测
- ✅ 实时调整相机角度

**注意：** 需要图形界面（不能在远程SSH会话中使用）

---

### 方法3：渲染为视频（推荐用于展示）

**导出为MP4视频文件**

```bash
python visualize_mujoco.py --mode video --output my_swing.mp4 --duration 5.0
```

**参数：**
- `--output`: 输出文件名（默认：`swing_simulation.mp4`）
- `--duration`: 仿真时长（秒，默认：3.0）

**输出：**
- 1280x720 分辨率
- 30 FPS
- H.264编码

**适用场景：**
- ✅ 制作演示视频
- ✅ 与用户分享结果
- ✅ 集成到前端展示
- ✅ 记录实验结果

---

## 📊 测试套件

快速测试所有可视化功能：

```bash
python test_visualization.py
```

**测试内容：**
1. ✅ matplotlib绘图功能
2. ✅ 图像渲染功能
3. ✅ 交互式查看器可用性

**输出文件：**
- `test_physics_plot.png` - 物理数据测试图
- `test_render.png` - 3D渲染测试图

---

## 🎨 高级用法

### 自定义相机角度

编辑 `visualize_mujoco.py` 中的相机设置：

```python
# 在 visualize_video() 或 visualize_interactive() 中修改
camera.lookat[:] = [0, 0, 1.0]  # 看向的点 (x, y, z)
camera.distance = 4.0            # 距离目标点的距离
camera.azimuth = 90              # 水平旋转角度（度）
camera.elevation = -15           # 垂直旋转角度（度）
```

**常用视角：**
- **正面视角**: `azimuth=0, elevation=0`
- **侧面视角**: `azimuth=90, elevation=-10`
- **俯视视角**: `azimuth=90, elevation=-45`
- **鸟瞰视角**: `azimuth=90, elevation=-75`

### 自定义初始姿势

修改关节角度来设置不同的起始姿势：

```python
# Address position (准备姿势)
data.qpos[2] = 1.0                      # 骨盆高度
data.qpos[7] = 20 * np.pi / 180         # 腰部旋转
data.qpos[11] = 90 * np.pi / 180        # 左肩抬起
data.qpos[14] = 45 * np.pi / 180        # 左肘弯曲
data.qpos[26] = 15 * np.pi / 180        # 左膝弯曲
```

**关节索引参考：**
- `qpos[0:3]` - 骨盆位置 (x, y, z)
- `qpos[3:7]` - 骨盆旋转 (四元数)
- `qpos[7]` - 腰部旋转
- `qpos[8]` - 腰部弯曲
- `qpos[11:17]` - 左臂关节
- `qpos[18:24]` - 右臂关节
- `qpos[25:30]` - 左腿关节
- `qpos[30:35]` - 右腿关节

---

## 🔍 故障排除

### 问题1：找不到matplotlib

```bash
pip install matplotlib
```

### 问题2：交互式查看器无法打开

**可能原因：**
- 在无图形界面的服务器上运行
- macOS需要额外配置

**解决方案：**
- 使用 `--mode plot` 或 `--mode video` 代替
- 或在本地机器上运行

### 问题3：视频渲染失败

```bash
# 安装视频编码依赖
pip install imageio imageio-ffmpeg
```

### 问题4：渲染速度慢

**优化建议：**
- 减少 `--duration` （默认3秒）
- 降低分辨率（编辑代码中的 `height` 和 `width`）
- 减少物理步数（增加 `steps_per_frame`）

---

## 💡 实用技巧

### 1. 批量生成多个视角的视频

```bash
# 正面视角
python visualize_mujoco.py --mode video --output front_view.mp4

# 修改代码中的 camera.azimuth = 0 后再运行
```

### 2. 保存高分辨率图像

在 `visualize_video()` 中修改：

```python
renderer = mujoco.Renderer(model, height=1080, width=1920)  # Full HD
```

### 3. 导出物理数据为CSV

在 `visualize_plot()` 末尾添加：

```python
import pandas as pd
df = pd.DataFrame({
    'time': times,
    'club_speed_mph': club_head_speeds,
    'pelvis_height_m': pelvis_heights
})
df.to_csv('physics_data.csv', index=False)
```

---

## 🚀 与Phase 2A集成

在实际的高尔夫分析中，可视化会集成到分析流程：

```python
# 1. AI提取姿态
pose_data = pose_estimator.forward(video_path)

# 2. MuJoCo仿真
simulator = MuJoCoSimulator("assets/mjcf/humanoid_golf.xml")
physics_data = simulator.forward(pose_data, output_path)

# 3. 可视化结果
# - 生成标注视频（骨骼点叠加）
# - 生成仿真动画（MuJoCo渲染）
# - 生成物理指标图表
visualize_analysis_results(physics_data)
```

---

## 📚 参考资料

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo Viewer Guide](https://mujoco.readthedocs.io/en/stable/programming.html#visualization)

---

## ✅ 检查清单

开发可视化功能时，确认以下内容：

- [ ] 模型加载正常（无错误）
- [ ] 关节运动范围合理（无穿模）
- [ ] 碰撞检测工作正常（脚与地面接触）
- [ ] 杆头速度在合理范围（50-150 mph）
- [ ] 视觉效果清晰（光照、材质）
- [ ] 输出文件格式正确（PNG/MP4）
