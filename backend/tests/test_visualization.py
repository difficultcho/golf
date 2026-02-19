"""
Simple MuJoCo visualization test

Tests different visualization methods
"""

import mujoco
import numpy as np
import time


def test_plot_physics():
    """Test plotting physics data"""
    print("📊 Testing physics data plotting...")

    # Check matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        print("  ✓ matplotlib available")
    except ImportError:
        print("  ✗ matplotlib not installed")
        return False

    # Load model
    model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
    data = mujoco.MjData(model)

    # Set initial pose
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)

    # Simulate and collect data
    times = []
    pelvis_heights = []
    lumbar_rotations = []

    print("  - Running 1 second simulation...")
    n_steps = 200  # 1 second

    for step in range(n_steps):
        mujoco.mj_step(model, data)

        if step % 5 == 0:
            times.append(data.time)
            pelvis_heights.append(data.qpos[2])
            lumbar_rotations.append(data.qpos[7] * 180 / np.pi)

    # Create simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(times, pelvis_heights, 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Pelvis Height')
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, lumbar_rotations, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Lumbar Rotation')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = 'test_physics_plot.png'
    plt.savefig(output_file, dpi=100)
    print(f"  ✓ Plot saved to {output_file}")

    return True


def test_render_image():
    """Test rendering to image"""
    print("\n🎨 Testing image rendering...")

    # Load model
    model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
    data = mujoco.MjData(model)

    # Set pose
    data.qpos[2] = 1.0
    data.qpos[11] = 90 * np.pi / 180  # Raise left arm
    data.qpos[18] = 90 * np.pi / 180  # Raise right arm
    mujoco.mj_forward(model, data)

    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Set camera
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 1.0]
    camera.distance = 3.0
    camera.azimuth = 90
    camera.elevation = -10

    # Render
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()

    print(f"  ✓ Rendered image: {pixels.shape}")

    # Save image
    try:
        import matplotlib.pyplot as plt
        plt.imsave('test_render.png', pixels)
        print(f"  ✓ Image saved to test_render.png")
        return True
    except Exception as e:
        print(f"  ✗ Failed to save image: {e}")
        return False


def test_interactive_viewer():
    """Test if interactive viewer can be launched"""
    print("\n🎮 Testing interactive viewer...")
    print("  Note: This requires a display. If running headless, this will fail.")

    try:
        import mujoco.viewer
        print("  ✓ mujoco.viewer module available")
        print("  ⚠️  Skipping actual viewer launch (would open window)")
        print("  To test interactively, run:")
        print("     python visualize_mujoco.py --mode viewer")
        return True
    except ImportError as e:
        print(f"  ✗ mujoco.viewer not available: {e}")
        return False


def main():
    print("="*60)
    print(" MuJoCo Visualization Test Suite")
    print("="*60)

    results = []

    # Test 1: Plot physics data
    results.append(("Physics Plotting", test_plot_physics()))

    # Test 2: Render image
    results.append(("Image Rendering", test_render_image()))

    # Test 3: Interactive viewer
    results.append(("Interactive Viewer", test_interactive_viewer()))

    # Summary
    print("\n" + "="*60)
    print(" Test Summary")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    total_passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
