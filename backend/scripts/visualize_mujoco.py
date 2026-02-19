"""
MuJoCo Golf Swing Visualization Tools

Provides multiple visualization options:
1. Interactive 3D viewer (real-time)
2. Render to video (for saving results)
3. Plot physics data (matplotlib charts)

Usage:
    # Interactive viewer
    python visualize_mujoco.py --mode viewer

    # Render to video
    python visualize_mujoco.py --mode video --output swing.mp4

    # Plot physics data
    python visualize_mujoco.py --mode plot
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path


def visualize_interactive():
    """
    Launch interactive 3D viewer

    Controls:
    - Mouse drag: Rotate camera
    - Scroll: Zoom
    - Right-click drag: Pan
    - Space: Pause/Resume
    - Backspace: Reset
    """
    print("🚀 Launching MuJoCo Interactive Viewer...")
    print("\nControls:")
    print("  - Mouse drag: Rotate camera")
    print("  - Scroll wheel: Zoom in/out")
    print("  - Right-click drag: Pan camera")
    print("  - Space: Pause/Resume simulation")
    print("  - Backspace: Reset to initial pose")
    print("  - ESC: Close viewer\n")

    # Load model
    model_path = "assets/mjcf/humanoid_golf.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Set initial pose (address position for golf)
    # Slightly bend knees and raise arms to hold club
    data.qpos[2] = 1.0  # Pelvis height
    data.qpos[7] = 20 * np.pi / 180   # Lumbar rotation (degrees to radians)
    data.qpos[9] = -10 * np.pi / 180  # Lumbar flexion

    # Left arm (holding club)
    data.qpos[11] = 90 * np.pi / 180   # Left shoulder flexion
    data.qpos[12] = 45 * np.pi / 180   # Left shoulder abduction
    data.qpos[14] = 45 * np.pi / 180   # Left elbow flexion

    # Right arm (holding club)
    data.qpos[18] = 90 * np.pi / 180   # Right shoulder flexion
    data.qpos[19] = -45 * np.pi / 180  # Right shoulder abduction
    data.qpos[21] = 45 * np.pi / 180   # Right elbow flexion

    # Legs (slight bend)
    data.qpos[26] = 15 * np.pi / 180   # Left knee
    data.qpos[31] = 15 * np.pi / 180   # Right knee

    # Forward kinematics to update positions
    mujoco.mj_forward(model, data)

    print("✓ Model loaded successfully")
    print(f"  - DOF: {model.nv}")
    print(f"  - Bodies: {model.nbody}")
    print("\n⏳ Starting simulation...\n")

    # Launch interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simulation loop
        while viewer.is_running():
            step_start = time.time()

            # Apply simple control: slight oscillation to demonstrate movement
            # In Phase 2A, this would be replaced with trajectory tracking
            t = data.time
            data.ctrl[0] = 0.3 * np.sin(2 * np.pi * 0.5 * t)  # Lumbar rotation

            # Step simulation
            mujoco.mj_step(model, data)

            # Sync viewer with simulation
            viewer.sync()

            # Maintain real-time speed (200 Hz)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def visualize_video(output_path: str = "swing_simulation.mp4", duration: float = 3.0):
    """
    Render simulation to video file

    Args:
        output_path: Path to save video
        duration: Simulation duration in seconds
    """
    print(f"🎥 Rendering MuJoCo simulation to video...")
    print(f"  - Output: {output_path}")
    print(f"  - Duration: {duration}s")

    # Load model
    model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
    data = mujoco.MjData(model)

    # Set initial pose (same as interactive)
    data.qpos[2] = 1.0
    data.qpos[7] = 20 * np.pi / 180
    data.qpos[11] = 90 * np.pi / 180
    data.qpos[12] = 45 * np.pi / 180
    data.qpos[14] = 45 * np.pi / 180
    data.qpos[18] = 90 * np.pi / 180
    data.qpos[19] = -45 * np.pi / 180
    data.qpos[21] = 45 * np.pi / 180
    data.qpos[26] = 15 * np.pi / 180
    data.qpos[31] = 15 * np.pi / 180

    mujoco.mj_forward(model, data)

    # Create renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)

    # Camera configuration
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 1.0]  # Look at pelvis height
    camera.distance = 4.0
    camera.azimuth = 90
    camera.elevation = -15

    # Simulate and render frames
    fps = 30
    n_frames = int(duration * fps)
    frames = []

    print(f"  - Rendering {n_frames} frames...")

    for frame_idx in range(n_frames):
        # Simulate multiple physics steps per frame
        steps_per_frame = int(1.0 / (fps * model.opt.timestep))
        for _ in range(steps_per_frame):
            # Apply control (oscillation for demo)
            t = data.time
            data.ctrl[0] = 0.5 * np.sin(2 * np.pi * 0.5 * t)
            mujoco.mj_step(model, data)

        # Render frame
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        frames.append(pixels.copy())

        if (frame_idx + 1) % 30 == 0:
            print(f"    Progress: {frame_idx + 1}/{n_frames} frames")

    # Save video using imageio
    try:
        import imageio
        print(f"  - Saving video to {output_path}...")
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"✓ Video saved successfully!")
        print(f"  - Resolution: 1280x720")
        print(f"  - FPS: {fps}")
        print(f"  - Frames: {n_frames}")
    except ImportError:
        print("⚠️  imageio not installed. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "imageio", "imageio-ffmpeg"])
        import imageio
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"✓ Video saved successfully!")


def visualize_plot():
    """
    Plot physics data from simulation
    """
    print("📊 Plotting physics data...")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt

    # Load model and run simulation
    model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
    data = mujoco.MjData(model)

    # Set initial pose
    data.qpos[2] = 1.0
    data.qpos[7] = 20 * np.pi / 180
    data.qpos[11] = 90 * np.pi / 180
    data.qpos[14] = 45 * np.pi / 180
    data.qpos[26] = 15 * np.pi / 180
    data.qpos[31] = 15 * np.pi / 180

    mujoco.mj_forward(model, data)

    # Storage for physics data
    times = []
    club_head_speeds = []
    pelvis_heights = []
    lumbar_rotations = []
    contact_forces = []

    # Get club head site ID
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_head_site")

    # Simulate for 3 seconds
    duration = 3.0
    n_steps = int(duration / model.opt.timestep)

    print(f"  - Simulating {duration}s...")

    for step in range(n_steps):
        # Apply control
        t = data.time
        data.ctrl[0] = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Lumbar rotation

        # Step simulation
        mujoco.mj_step(model, data)
        mujoco.mj_inverse(model, data)

        # Record data every 10 steps (reduce data points)
        if step % 10 == 0:
            times.append(data.time)

            # Club head speed (from sensor data)
            # The framelinvel sensor for club_head_site is sensor index 5
            sensor_adr = model.sensor_adr[5]  # club_head_vel sensor
            club_vel = data.sensordata[sensor_adr:sensor_adr+3]
            speed_ms = np.linalg.norm(club_vel)
            speed_mph = speed_ms * 2.23694
            club_head_speeds.append(speed_mph)

            # Pelvis height
            pelvis_heights.append(data.qpos[2])

            # Lumbar rotation (convert to degrees)
            lumbar_rotations.append(data.qpos[7] * 180 / np.pi)

            # Contact forces
            total_force = 0
            for i in range(data.ncon):
                contact = data.contact[i]
                # Get contact force magnitude from constraint forces
                total_force += np.linalg.norm(data.efc_force[i*6:(i+1)*6]) if i*6 < len(data.efc_force) else 0
            contact_forces.append(total_force)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MuJoCo Golf Swing Physics Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Club head speed
    axes[0, 0].plot(times, club_head_speeds, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Speed (mph)')
    axes[0, 0].set_title('Club Head Speed')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=np.mean(club_head_speeds), color='r', linestyle='--',
                       label=f'Mean: {np.mean(club_head_speeds):.1f} mph')
    axes[0, 0].legend()

    # Plot 2: Pelvis height
    axes[0, 1].plot(times, pelvis_heights, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Height (m)')
    axes[0, 1].set_title('Pelvis Height (Balance)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Initial: 1.0m')
    axes[0, 1].legend()

    # Plot 3: Lumbar rotation
    axes[1, 0].plot(times, lumbar_rotations, 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Angle (degrees)')
    axes[1, 0].set_title('Lumbar Rotation (X-Factor)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Plot 4: Contact forces
    axes[1, 1].plot(times, contact_forces, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Force (N)')
    axes[1, 1].set_title('Ground Reaction Force')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = "physics_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")

    # Show plot
    print("  - Displaying plot (close window to continue)...")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='MuJoCo Golf Swing Visualizer')
    parser.add_argument('--mode', type=str, default='viewer',
                        choices=['viewer', 'video', 'plot'],
                        help='Visualization mode')
    parser.add_argument('--output', type=str, default='swing_simulation.mp4',
                        help='Output path for video mode')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Simulation duration in seconds (for video/plot)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print(" MuJoCo Golf Swing Visualizer")
    print("="*60 + "\n")

    if args.mode == 'viewer':
        visualize_interactive()
    elif args.mode == 'video':
        visualize_video(args.output, args.duration)
    elif args.mode == 'plot':
        visualize_plot()

    print("\n✓ Visualization complete!\n")


if __name__ == "__main__":
    main()
