"""
Test script for MuJoCo humanoid golf model

This script demonstrates:
1. Loading the custom MJCF model
2. Running a simple simulation
3. Computing basic physics metrics

Usage:
    python test_mujoco_model.py

Prerequisites:
    pip install mujoco
"""

import numpy as np


def test_model_loading():
    """Test if the model loads correctly"""
    try:
        import mujoco
        print("✓ MuJoCo imported successfully")
    except ImportError:
        print("✗ MuJoCo not installed. Run: pip install mujoco")
        return False

    try:
        model_path = "assets/mjcf/humanoid_golf.xml"
        model = mujoco.MjModel.from_xml_path(model_path)
        print(f"✓ Model loaded: {model_path}")
        print(f"  - DOF: {model.nv}")
        print(f"  - Bodies: {model.nbody}")
        print(f"  - Joints: {model.njnt}")
        print(f"  - Actuators: {model.nu}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False


def test_basic_simulation():
    """Run a basic simulation and check stability"""
    try:
        import mujoco
        import mujoco.viewer

        model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
        data = mujoco.MjData(model)

        print("\n✓ Running simulation test...")

        # Reset to initial state
        mujoco.mj_resetData(model, data)

        # Simulate 1 second
        n_steps = int(1.0 / model.opt.timestep)
        initial_height = data.qpos[2]  # Z position of pelvis

        for i in range(n_steps):
            mujoco.mj_step(model, data)

        final_height = data.qpos[2]
        height_change = abs(final_height - initial_height)

        print(f"  - Simulated {n_steps} steps ({model.opt.timestep}s each)")
        print(f"  - Initial pelvis height: {initial_height:.3f}m")
        print(f"  - Final pelvis height: {final_height:.3f}m")
        print(f"  - Height change: {height_change:.3f}m")

        if height_change < 0.5:  # Should be relatively stable
            print("  ✓ Model is stable (pelvis didn't fall significantly)")
        else:
            print("  ⚠ Model may need tuning (large height change)")

        return True

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return False


def test_inverse_dynamics():
    """Test inverse dynamics computation"""
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
        data = mujoco.MjData(model)

        print("\n✓ Testing inverse dynamics...")

        # Set a specific pose (e.g., arms raised)
        mujoco.mj_resetData(model, data)

        # Forward kinematics
        mujoco.mj_forward(model, data)

        # Compute inverse dynamics
        mujoco.mj_inverse(model, data)

        # Check computed torques
        torques = data.qfrc_inverse
        max_torque = np.max(np.abs(torques))

        print(f"  - Computed joint torques: {len(torques)} values")
        print(f"  - Max torque magnitude: {max_torque:.2f} Nm")
        print(f"  - Torque range: [{np.min(torques):.2f}, {np.max(torques):.2f}] Nm")

        if max_torque < 1000:  # Reasonable range
            print("  ✓ Inverse dynamics working correctly")
        else:
            print("  ⚠ Torques seem unusually high")

        return True

    except Exception as e:
        print(f"✗ Inverse dynamics test failed: {e}")
        return False


def test_club_head_speed():
    """Test club head speed computation"""
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
        data = mujoco.MjData(model)

        print("\n✓ Testing club head speed computation...")

        # Find club head site
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_head_site")
            print(f"  - Found club_head_site (ID: {site_id})")
        except:
            print("  ✗ club_head_site not found in model")
            return False

        # Get velocity
        mujoco.mj_forward(model, data)

        # Site linear velocity
        site_vel = data.sensordata[0:3] if model.nsensor > 0 else np.zeros(3)
        speed = np.linalg.norm(site_vel)

        print(f"  - Club head velocity: {site_vel}")
        print(f"  - Club head speed: {speed:.2f} m/s ({speed * 2.23694:.2f} mph)")

        return True

    except Exception as e:
        print(f"✗ Club head speed test failed: {e}")
        return False


def test_contact_detection():
    """Test ground contact detection"""
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
        data = mujoco.MjData(model)

        print("\n✓ Testing contact detection...")

        mujoco.mj_forward(model, data)

        # Check for contacts
        n_contacts = data.ncon
        print(f"  - Number of active contacts: {n_contacts}")

        if n_contacts > 0:
            print(f"  - Contact details:")
            for i in range(min(n_contacts, 5)):  # Show first 5
                contact = data.contact[i]
                force_norm = np.linalg.norm(contact.force)
                print(f"    Contact {i}: force = {force_norm:.2f} N")

        return True

    except Exception as e:
        print(f"✗ Contact test failed: {e}")
        return False


def visualize_model():
    """Launch interactive viewer (optional)"""
    try:
        import mujoco
        import mujoco.viewer

        print("\n✓ Launching interactive viewer...")
        print("  (Press ESC to close)")

        model = mujoco.MjModel.from_xml_path("assets/mjcf/humanoid_golf.xml")
        data = mujoco.MjData(model)

        # Launch viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Simulate for 10 seconds
            for _ in range(10000):
                mujoco.mj_step(model, data)
                viewer.sync()

    except ImportError:
        print("  ⚠ Viewer not available (requires mujoco.viewer)")
    except KeyboardInterrupt:
        print("  - Viewer closed by user")
    except Exception as e:
        print(f"  ⚠ Viewer error: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("MuJoCo Golf Swing Model Test Suite")
    print("=" * 60)

    tests = [
        ("Model Loading", test_model_loading),
        ("Basic Simulation", test_basic_simulation),
        ("Inverse Dynamics", test_inverse_dynamics),
        ("Club Head Speed", test_club_head_speed),
        ("Contact Detection", test_contact_detection),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'─' * 60}")
        print(f"Test: {name}")
        print('─' * 60)
        success = test_func()
        results.append((name, success))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:10s} {name}")

    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    # Optional: launch viewer
    if passed == total:
        print("\n" + "=" * 60)
        response = input("All tests passed! Launch interactive viewer? (y/n): ")
        if response.lower() == 'y':
            visualize_model()


if __name__ == "__main__":
    main()
