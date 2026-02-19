"""
Demo: MuJoCo可视化在高尔夫分析中的实际应用

演示如何使用可视化工具来验证和展示仿真结果
"""

import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt
from pathlib import Path


def simulate_golf_swing(duration=2.0):
    """
    模拟一个简化的高尔夫挥杆动作

    在实际Phase 2A中，这个函数会被替换为：
    - AI姿态估计的输出
    - mocap驱动的轨迹

    Returns:
        dict: 包含仿真数据的字典
    """
    print("⛳ 模拟高尔夫挥杆动作...")

    # 加载模型
    model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
    data = mujoco.MjData(model)

    # 设置初始姿势（Address position）
    data.qpos[2] = 1.0  # 站立高度
    data.qpos[7] = 20 * np.pi / 180   # 腰部轻微旋转
    data.qpos[11] = 100 * np.pi / 180  # 左臂抬起
    data.qpos[14] = 30 * np.pi / 180   # 左肘弯曲
    data.qpos[18] = 100 * np.pi / 180  # 右臂抬起
    data.qpos[21] = 30 * np.pi / 180   # 右肘弯曲
    data.qpos[26] = 20 * np.pi / 180   # 左膝微曲
    data.qpos[31] = 20 * np.pi / 180   # 右膝微曲

    mujoco.mj_forward(model, data)

    # 获取传感器ID
    club_head_sensor_id = 5  # framelinvel sensor for club_head_site

    # 存储数据
    times = []
    club_speeds = []
    pelvis_heights = []
    lumbar_rotations = []
    left_knee_angles = []
    contact_forces_left = []
    contact_forces_right = []

    # 挥杆阶段定义
    phases = []
    current_phase = "Address"

    # 仿真参数
    n_steps = int(duration / model.opt.timestep)

    print(f"  - 仿真时长: {duration}s")
    print(f"  - 时间步长: {model.opt.timestep}s")
    print(f"  - 总步数: {n_steps}")
    print(f"  - 仿真中...")

    for step in range(n_steps):
        t = data.time

        # 控制策略：模拟挥杆的3个阶段
        # Phase 1: Backswing (0-0.8s) - 上杆
        if t < 0.8:
            current_phase = "Backswing"
            # 腰部向右旋转，抬起球杆
            target_lumbar_rotation = 60 * np.pi / 180 * (t / 0.8)
            data.ctrl[0] = (target_lumbar_rotation - data.qpos[7]) * 100  # PD控制

        # Phase 2: Downswing (0.8-1.2s) - 下杆
        elif t < 1.2:
            current_phase = "Downswing"
            # 快速向左旋转，加速球杆
            progress = (t - 0.8) / 0.4
            target_lumbar_rotation = 60 * (1 - progress) - 30 * progress
            data.ctrl[0] = (target_lumbar_rotation * np.pi / 180 - data.qpos[7]) * 150

        # Phase 3: Follow-through (1.2s+) - 随挥
        else:
            current_phase = "Follow-through"
            # 继续旋转到终点
            target_lumbar_rotation = -40 * np.pi / 180
            data.ctrl[0] = (target_lumbar_rotation - data.qpos[7]) * 80

        # 执行仿真步
        mujoco.mj_step(model, data)
        mujoco.mj_inverse(model, data)

        # 每10步记录一次数据
        if step % 10 == 0:
            times.append(t)
            phases.append(current_phase)

            # 杆头速度
            sensor_adr = model.sensor_adr[club_head_sensor_id]
            club_vel = data.sensordata[sensor_adr:sensor_adr+3]
            speed_ms = np.linalg.norm(club_vel)
            speed_mph = speed_ms * 2.23694
            club_speeds.append(speed_mph)

            # 骨盆高度
            pelvis_heights.append(data.qpos[2])

            # 腰部旋转
            lumbar_rotations.append(data.qpos[7] * 180 / np.pi)

            # 左膝角度
            left_knee_angles.append(data.qpos[26] * 180 / np.pi)

            # 接触力（从传感器获取）
            left_foot_sensor_adr = model.sensor_adr[2]  # left_foot_contact
            right_foot_sensor_adr = model.sensor_adr[3]  # right_foot_contact
            contact_forces_left.append(data.sensordata[left_foot_sensor_adr])
            contact_forces_right.append(data.sensordata[right_foot_sensor_adr])

    print("  ✓ 仿真完成")

    return {
        'model': model,
        'data': data,
        'times': times,
        'club_speeds': club_speeds,
        'pelvis_heights': pelvis_heights,
        'lumbar_rotations': lumbar_rotations,
        'left_knee_angles': left_knee_angles,
        'contact_forces_left': contact_forces_left,
        'contact_forces_right': contact_forces_right,
        'phases': phases
    }


def visualize_comprehensive_analysis(sim_data):
    """
    生成完整的物理分析可视化报告
    """
    print("\n📊 生成综合分析报告...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    times = sim_data['times']

    # 1. 杆头速度曲线
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, sim_data['club_speeds'], 'b-', linewidth=2.5, label='Club Head Speed')
    ax1.fill_between(times, 0, sim_data['club_speeds'], alpha=0.3)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Speed (mph)', fontsize=11)
    ax1.set_title('Club Head Speed Analysis', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 标记峰值速度
    max_speed = max(sim_data['club_speeds'])
    max_idx = sim_data['club_speeds'].index(max_speed)
    max_time = times[max_idx]
    ax1.plot(max_time, max_speed, 'ro', markersize=10)
    ax1.annotate(f'Peak: {max_speed:.1f} mph\n@ {max_time:.2f}s',
                xy=(max_time, max_speed), xytext=(max_time+0.2, max_speed-5),
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # 2. 统计摘要
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    summary_text = f"""
    📈 Performance Metrics

    Peak Speed: {max_speed:.1f} mph
    Impact Time: {max_time:.2f} s

    Average Speed: {np.mean(sim_data['club_speeds']):.1f} mph

    Acceleration:
      Max: {max(np.diff(sim_data['club_speeds'])):.1f} mph/frame

    Swing Duration: {times[-1]:.2f} s
    """
    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. 骨盆高度（平衡分析）
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(times, sim_data['pelvis_heights'], 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Height (m)', fontsize=10)
    ax3.set_title('Balance (Pelvis Height)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Initial')
    ax3.legend()

    # 4. 腰部旋转（X-Factor）
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(times, sim_data['lumbar_rotations'], 'orange', linewidth=2)
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Angle (degrees)', fontsize=10)
    ax4.set_title('X-Factor (Lumbar Rotation)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # 标注最大旋转角
    max_rotation = max(sim_data['lumbar_rotations'], key=abs)
    ax4.axhline(y=max_rotation, color='r', linestyle='--', alpha=0.5,
               label=f'Max: {abs(max_rotation):.1f}°')
    ax4.legend()

    # 5. 左膝角度
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(times, sim_data['left_knee_angles'], 'purple', linewidth=2)
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Angle (degrees)', fontsize=10)
    ax5.set_title('Left Knee Flexion', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. 地面反作用力（双脚）
    ax6 = fig.add_subplot(gs[2, :])
    ax6.plot(times, sim_data['contact_forces_left'], 'b-', linewidth=2, label='Left Foot', alpha=0.7)
    ax6.plot(times, sim_data['contact_forces_right'], 'r-', linewidth=2, label='Right Foot', alpha=0.7)
    ax6.fill_between(times, 0, sim_data['contact_forces_left'], color='b', alpha=0.2)
    ax6.fill_between(times, 0, sim_data['contact_forces_right'], color='r', alpha=0.2)
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Force (N)', fontsize=11)
    ax6.set_title('Ground Reaction Forces', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper right', fontsize=10)

    # 添加挥杆阶段标注
    phase_changes = []
    for i in range(1, len(sim_data['phases'])):
        if sim_data['phases'][i] != sim_data['phases'][i-1]:
            phase_changes.append((times[i], sim_data['phases'][i]))

    for t, phase in phase_changes:
        ax6.axvline(x=t, color='gray', linestyle=':', alpha=0.5)
        ax6.text(t, ax6.get_ylim()[1]*0.9, phase, rotation=90,
                verticalalignment='top', fontsize=9, alpha=0.7)

    # 总标题
    fig.suptitle('MuJoCo Golf Swing Physics Analysis Report',
                fontsize=16, fontweight='bold', y=0.98)

    # 保存
    output_file = 'golf_analysis_report.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ 报告已保存: {output_file}")

    return output_file


def render_swing_frames(sim_data, n_frames=8):
    """
    渲染挥杆的关键帧
    """
    print("\n🎬 渲染挥杆关键帧...")

    model = sim_data['model']
    data = sim_data['data']

    # 重新仿真到不同的时间点
    renderer = mujoco.Renderer(model, height=400, width=400)

    # 相机设置
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 1.0]
    camera.distance = 3.5
    camera.azimuth = 110
    camera.elevation = -15

    # 创建画布
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    duration = sim_data['times'][-1]
    frame_times = np.linspace(0, duration, n_frames)

    for idx, target_time in enumerate(frame_times):
        # 重置并仿真到目标时间
        mujoco.mj_resetData(model, data)

        # 恢复初始姿势
        data.qpos[2] = 1.0
        data.qpos[7] = 20 * np.pi / 180
        data.qpos[11] = 100 * np.pi / 180
        data.qpos[14] = 30 * np.pi / 180
        data.qpos[18] = 100 * np.pi / 180
        data.qpos[21] = 30 * np.pi / 180
        data.qpos[26] = 20 * np.pi / 180
        data.qpos[31] = 20 * np.pi / 180

        # 仿真到目标时间（简化：使用相同控制）
        while data.time < target_time:
            t = data.time
            if t < 0.8:
                target_lumbar = 60 * np.pi / 180 * (t / 0.8)
            elif t < 1.2:
                progress = (t - 0.8) / 0.4
                target_lumbar = (60 * (1 - progress) - 30 * progress) * np.pi / 180
            else:
                target_lumbar = -40 * np.pi / 180
            data.ctrl[0] = (target_lumbar - data.qpos[7]) * 100
            mujoco.mj_step(model, data)

        # 渲染
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()

        # 显示
        axes[idx].imshow(pixels)
        axes[idx].axis('off')
        axes[idx].set_title(f't = {target_time:.2f}s', fontsize=10)

        print(f"  - 帧 {idx+1}/{n_frames}: {target_time:.2f}s")

    plt.suptitle('Golf Swing Key Frames (MuJoCo Simulation)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = 'swing_keyframes.png'
    plt.savefig(output_file, dpi=120, bbox_inches='tight')
    print(f"  ✓ 关键帧已保存: {output_file}")

    return output_file


def main():
    """主演示流程"""
    print("="*70)
    print(" MuJoCo可视化演示 - 高尔夫挥杆分析")
    print("="*70)

    # 1. 运行仿真
    sim_data = simulate_golf_swing(duration=2.0)

    # 2. 生成综合分析报告
    report_file = visualize_comprehensive_analysis(sim_data)

    # 3. 渲染关键帧
    keyframes_file = render_swing_frames(sim_data)

    # 总结
    print("\n" + "="*70)
    print(" 演示完成！")
    print("="*70)
    print("\n生成的文件：")
    print(f"  1. {report_file} - 物理分析报告（6个子图）")
    print(f"  2. {keyframes_file} - 挥杆关键帧（8帧）")
    print("\n💡 提示：")
    print("  - 在实际Phase 2A中，控制信号会被mocap轨迹替换")
    print("  - 这些可视化工具可用于验证AI姿态估计的准确性")
    print("  - 物理指标可作为改进建议的依据\n")


if __name__ == "__main__":
    main()
