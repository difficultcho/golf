"""
MuJoCoSimulator - Physics simulation using MuJoCo

This module reconstructs golf swing motion in MuJoCo physics simulator
using a PD controller to track target joint angles derived from pose data,
then extracts biomechanics metrics via inverse dynamics.

PD Controller Theory:
    u(t) = Kp * (q_target - q_current) - Kd * q_velocity
    ctrl[i] = clip(u[i] / gear[i], -1, 1)

    Kp (proportional gain): determines response strength to position error
    Kd (derivative gain): provides damping to prevent oscillation
"""

import torch
import torch.nn as nn
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy.signal import savgol_filter

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: MuJoCo not installed. MuJoCoSimulator will use dummy data.")


class MuJoCoSimulator(nn.Module):
    """
    Physics simulation using MuJoCo with PD control.

    Converts MediaPipe pose keypoints to joint angles via geometric IK,
    then drives the humanoid with a PD controller to track the target
    trajectory. Extracts joint torques, contact forces, and COM data.
    """

    # PD controller gains (Nm/rad and Nm·s/rad)
    # Increase Kp for stiffer tracking, increase Kd to reduce oscillation
    # Legs need higher gains to support body weight against gravity
    KP_DEFAULT = 200.0
    KD_DEFAULT = 20.0
    KP_LEGS = 500.0   # higher stiffness for hip/knee/ankle
    KD_LEGS = 50.0

    def __init__(self, mjcf_path: str):
        super().__init__()
        self.version = nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.mjcf_path = mjcf_path

        if MUJOCO_AVAILABLE and Path(mjcf_path).exists():
            try:
                self.model = mujoco.MjModel.from_xml_path(mjcf_path)
                self.data = mujoco.MjData(self.model)
                self.mujoco_ready = True
                print(f"MuJoCo model loaded: {mjcf_path}")
                print(f"  - nq (qpos size): {self.model.nq}")
                print(f"  - nv (velocity DOF): {self.model.nv}")
                print(f"  - nu (actuators): {self.model.nu}")
                print(f"  - nbody: {self.model.nbody}")
                self._build_actuator_lookup()
            except Exception as e:
                print(f"Warning: Failed to load MuJoCo model: {e}")
                self.mujoco_ready = False
                self.model = None
                self.data = None
        else:
            self.mujoco_ready = False
            self.model = None
            self.data = None

            if not MUJOCO_AVAILABLE:
                print("MuJoCo not available")
            if not Path(mjcf_path).exists():
                print(f"MJCF file not found: {mjcf_path}")

    def _build_actuator_lookup(self):
        """
        Build lookup tables mapping actuator index to qpos/qvel addresses.

        This is necessary because the ball joint (club_grip) inserts extra
        qpos entries that shift all subsequent joint indices. Using
        model.jnt_qposadr and model.jnt_dofadr gives the correct addresses.
        """
        nu = self.model.nu
        self._act_qpos_adr = np.zeros(nu, dtype=int)
        self._act_dof_adr = np.zeros(nu, dtype=int)
        self._act_gear = np.zeros(nu)
        self._joint_limits = np.zeros((nu, 2))
        self._act_name_to_idx = {}

        print("  Actuator lookup table:")
        for i in range(nu):
            joint_id = self.model.actuator_trnid[i, 0]
            self._act_qpos_adr[i] = self.model.jnt_qposadr[joint_id]
            self._act_dof_adr[i] = self.model.jnt_dofadr[joint_id]
            self._act_gear[i] = self.model.actuator_gear[i, 0]
            self._joint_limits[i] = self.model.jnt_range[joint_id]

            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._act_name_to_idx[name] = i
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            print(f"    [{i:2d}] {name or '?':30s} -> joint '{joint_name}' "
                  f"qpos={self._act_qpos_adr[i]:2d} dof={self._act_dof_adr[i]:2d} "
                  f"gear={self._act_gear[i]:.0f}")

        # Per-actuator PD gains: legs get higher stiffness
        leg_keywords = {"hip", "knee", "ankle"}
        self._kp = np.full(nu, self.KP_DEFAULT)
        self._kd = np.full(nu, self.KD_DEFAULT)
        for i in range(nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if any(kw in name for kw in leg_keywords):
                self._kp[i] = self.KP_LEGS
                self._kd[i] = self.KD_LEGS

        # Touch sensor addresses for contact force extraction
        self._left_foot_sensor_adr = self._find_sensor_adr("left_foot_contact")
        self._right_foot_sensor_adr = self._find_sensor_adr("right_foot_contact")
        # Club head velocity sensor
        self._club_vel_sensor_adr = self._find_sensor_adr("club_head_vel")

        print(f"  Touch sensors: left_foot={self._left_foot_sensor_adr}, "
              f"right_foot={self._right_foot_sensor_adr}")
        print(f"  Club velocity sensor: adr={self._club_vel_sensor_adr}")

    def _find_sensor_adr(self, sensor_name: str) -> int:
        """Find sensor data address by name. Returns -1 if not found."""
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id < 0:
            return -1
        return self.model.sensor_adr[sensor_id]

    def forward(self, pose_data_path: Path, output_path: Path) -> Dict:
        """
        Replay trajectory in MuJoCo and extract physics data.

        Args:
            pose_data_path: Path to JSON file with pose keypoints
            output_path: Path to save physics data JSON

        Returns:
            Dictionary with physics data
        """
        if not self.mujoco_ready or self.model is None:
            return self._generate_dummy_physics_data(pose_data_path, output_path)

        start_time = time.time()

        with open(pose_data_path, 'r') as f:
            pose_data = json.load(f)

        joints_3d = pose_data['joints_3d']  # (T, 17, 3)
        timestamps = pose_data['timestamps']
        fps = pose_data.get('fps', 30.0)

        # Convert pose keypoints to target joint angles
        joint_angles_sequence = self._map_pose_to_joint_angles(joints_3d)

        # Smooth trajectory to reduce jitter from pose estimation
        joint_angles_sequence = self._smooth_trajectory(joint_angles_sequence)

        # Run PD-controlled simulation
        physics_data = self._simulate_trajectory(joint_angles_sequence, fps)

        with open(output_path, 'w') as f:
            json.dump(physics_data, f, indent=2)

        processing_time = time.time() - start_time

        return {
            "joint_angles": physics_data["joint_angles"],
            "joint_velocities": physics_data["joint_velocities"],
            "joint_torques": physics_data["joint_torques"],
            "contact_forces": physics_data["contact_forces"],
            "com_position": physics_data["com_position"],
            "duration": len(timestamps) / fps if fps > 0 else 0.0,
            "frame_count": len(timestamps),
            "processing_time": processing_time,
            "model_version": f"mujoco_sim_v{self.version.item()}"
        }

    # ── Geometric Inverse Kinematics ──────────────────────────────────

    def _map_pose_to_joint_angles(self, joints_3d: List[List[List[float]]]) -> np.ndarray:
        """
        Map MediaPipe COCO-17 keypoints to MuJoCo target joint angles.

        Uses geometric IK: computes angles from limb vectors via atan2.
        This is a heuristic approximation — proper IK would iteratively
        solve FK(q) = target, but geometric IK is sufficient for
        trajectory tracking with a PD controller.

        Args:
            joints_3d: (T, 17, 3) keypoints in MediaPipe normalized image coords
                       x: 0-1 left-right, y: 0-1 top-bottom, z: depth

        Returns:
            Joint angles array (T, nu) in radians, one per actuator
        """
        n_frames = len(joints_3d)
        nu = self.model.nu
        joint_angles = np.zeros((n_frames, nu))

        for t in range(n_frames):
            joint_angles[t] = self._compute_joint_angles_from_pose(
                np.array(joints_3d[t]))

        if n_frames > 0:
            print(f"  IK: frame 0 target angles (deg): "
                  f"{np.degrees(joint_angles[0][:6]).round(1).tolist()} ...")

        return joint_angles

    def _compute_joint_angles_from_pose(self, kp: np.ndarray) -> np.ndarray:
        """
        Compute target angles for all actuated joints from a single frame.

        Args:
            kp: (17, 3) COCO keypoints in MediaPipe normalized coords

        Returns:
            (nu,) array of target joint angles in radians
        """
        nu = self.model.nu
        angles = np.zeros(nu)

        # ── Step 1: Coordinate transform ──
        # MediaPipe image coords: x=right (0-1), y=DOWN (0-1), z=depth
        # After transform: x=left/right, y=UP, z=forward/backward (depth)
        pts = kp.copy()
        pts[:, 1] = -pts[:, 1]  # flip y so up is positive

        # Center on hip midpoint
        hip_center = (pts[11] + pts[12]) / 2.0
        pts = pts - hip_center

        # Scale by shoulder width → approximate meters (shoulder width ~0.4m)
        shoulder_width = np.linalg.norm(pts[5] - pts[6])
        if shoulder_width > 1e-4:
            scale = 0.4 / shoulder_width
        else:
            scale = 1.0
        pts = pts * scale

        # Extract keypoints
        l_shoulder = pts[5]
        r_shoulder = pts[6]
        l_elbow = pts[7]
        r_elbow = pts[8]
        l_wrist = pts[9]
        r_wrist = pts[10]
        l_hip = pts[11]
        r_hip = pts[12]
        l_knee = pts[13]
        r_knee = pts[14]
        l_ankle = pts[15]
        r_ankle = pts[16]

        shoulder_mid = (l_shoulder + r_shoulder) / 2.0
        hip_mid = (l_hip + r_hip) / 2.0  # ~origin after centering

        # ── Step 2: Compute joint angles ──
        # After transform: y is UP (vertical), z is depth, x is left/right

        def flexion_angle(vec):
            """Angle from vertical (-y) in sagittal plane (y-z).
            Positive = forward (toward camera, -z direction).
            A straight-down limb has flexion=0."""
            return np.arctan2(-vec[2], -vec[1]) if abs(vec[1]) + abs(vec[2]) > 1e-6 else 0.0

        def abduction_angle(vec):
            """Angle from vertical (-y) in frontal plane (x-y).
            Positive = outward (away from body midline).
            A straight-down limb has abduction=0."""
            return np.arctan2(vec[0], -vec[1]) if abs(vec[1]) + abs(vec[0]) > 1e-6 else 0.0

        # Helper: angle between two vectors
        def vec_angle(v1, v2):
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return 0.0
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            return np.arccos(cos_a)

        # Torso vector (hip midpoint → shoulder midpoint), should point upward
        torso_vec = shoulder_mid - hip_mid

        # --- Lumbar rotation: shoulder line vs hip line in horizontal (x-z) plane ---
        shoulder_dir = l_shoulder - r_shoulder  # left-to-right
        hip_dir = l_hip - r_hip
        a_shoulder = np.arctan2(shoulder_dir[2], shoulder_dir[0])
        a_hip = np.arctan2(hip_dir[2], hip_dir[0])
        lumbar_rot = a_shoulder - a_hip
        self._set_angle(angles, "lumbar_rotation_motor", lumbar_rot)

        # --- Lumbar flexion: torso tilt from vertical in sagittal (y-z) plane ---
        # Torso pointing straight up → flexion=0
        # Torso leaning forward → positive flexion
        lumbar_flex = np.arctan2(-torso_vec[2], torso_vec[1]) \
            if abs(torso_vec[1]) + abs(torso_vec[2]) > 1e-6 else 0.0
        self._set_angle(angles, "lumbar_flexion_motor", lumbar_flex)

        # --- Left arm ---
        l_upper_arm = l_elbow - l_shoulder  # shoulder → elbow
        l_forearm = l_wrist - l_elbow       # elbow → wrist

        l_shoulder_flex = flexion_angle(l_upper_arm)
        self._set_angle(angles, "left_shoulder_flex_motor", l_shoulder_flex)

        l_shoulder_abd = abduction_angle(l_upper_arm)
        self._set_angle(angles, "left_shoulder_abd_motor", l_shoulder_abd)

        # Elbow flexion: 0° = straight, 145° = fully bent
        l_elbow_angle = vec_angle(l_upper_arm, l_forearm)
        self._set_angle(angles, "left_elbow_flex_motor", l_elbow_angle)

        # --- Right arm ---
        r_upper_arm = r_elbow - r_shoulder
        r_forearm = r_wrist - r_elbow

        r_shoulder_flex = flexion_angle(r_upper_arm)
        self._set_angle(angles, "right_shoulder_flex_motor", r_shoulder_flex)

        r_shoulder_abd = abduction_angle(r_upper_arm)
        self._set_angle(angles, "right_shoulder_abd_motor", r_shoulder_abd)

        r_elbow_angle = vec_angle(r_upper_arm, r_forearm)
        self._set_angle(angles, "right_elbow_flex_motor", r_elbow_angle)

        # --- Left leg ---
        l_thigh = l_knee - l_hip    # hip → knee
        l_shin = l_ankle - l_knee   # knee → ankle

        l_hip_flex = flexion_angle(l_thigh)
        self._set_angle(angles, "left_hip_flex_motor", l_hip_flex)

        l_hip_abd = abduction_angle(l_thigh)
        self._set_angle(angles, "left_hip_abd_motor", l_hip_abd)

        # Knee flexion: 0° = straight, 135° = fully bent
        l_knee_angle = vec_angle(l_thigh, l_shin)
        self._set_angle(angles, "left_knee_flex_motor", l_knee_angle)

        # --- Right leg ---
        r_thigh = r_knee - r_hip
        r_shin = r_ankle - r_knee

        r_hip_flex = flexion_angle(r_thigh)
        self._set_angle(angles, "right_hip_flex_motor", r_hip_flex)

        r_hip_abd = abduction_angle(r_thigh)
        self._set_angle(angles, "right_hip_abd_motor", r_hip_abd)

        r_knee_angle = vec_angle(r_thigh, r_shin)
        self._set_angle(angles, "right_knee_flex_motor", r_knee_angle)

        # Joints with insufficient COCO data: shoulder rotation, wrist, ankle,
        # hip rotation, neck — leave at 0 (neutral pose)

        return angles

    def _set_angle(self, angles: np.ndarray, actuator_name: str, value: float):
        """Set a target angle by actuator name, clamped to joint limits."""
        idx = self._act_name_to_idx.get(actuator_name)
        if idx is not None:
            lo, hi = self._joint_limits[idx]
            angles[idx] = np.clip(value, lo, hi)

    # ── Trajectory Smoothing ─────────────────────────────────────────

    def _smooth_trajectory(self, trajectory: np.ndarray,
                           window_length: int = 11, polyorder: int = 3) -> np.ndarray:
        """
        Smooth trajectory using Savitzky-Golay filter to reduce
        jitter from per-frame pose estimation.
        """
        if len(trajectory) < window_length:
            return trajectory

        smoothed = np.zeros_like(trajectory)
        for j in range(trajectory.shape[1]):
            smoothed[:, j] = savgol_filter(trajectory[:, j], window_length, polyorder)

        return smoothed

    # ── PD-Controlled Simulation ─────────────────────────────────────

    def _simulate_trajectory(self, joint_angles: np.ndarray, fps: float) -> Dict:
        """
        Run MuJoCo simulation with PD controller tracking target angles.

        PD Control Law:
            torque = Kp * (q_target - q_current) - Kd * q_velocity
            ctrl[i] = clip(torque / gear[i], -1, 1)

        The simulation sub-steps within each video frame to match
        MuJoCo's physics timestep (0.005s) with the video framerate (~30fps).

        Args:
            joint_angles: Target angles (T, nu) in radians
            fps: Video frames per second

        Returns:
            Dictionary with simulation results
        """
        n_frames = joint_angles.shape[0]
        nu = self.model.nu

        # Sub-stepping: how many physics steps per video frame
        frame_dt = 1.0 / fps
        sim_dt = self.model.opt.timestep  # typically 0.005s
        steps_per_frame = max(1, int(round(frame_dt / sim_dt)))
        print(f"  Simulation: {n_frames} frames, {steps_per_frame} sub-steps/frame "
              f"(video {fps:.0f}fps, physics {1/sim_dt:.0f}Hz)")

        # Storage
        all_joint_angles = []
        all_joint_velocities = []
        all_joint_torques = []
        all_contact_forces = []
        all_com_positions = []

        # Reset simulation, lower pelvis so feet touch ground
        # Default qpos z=1.0 leaves feet 8cm above ground; z=0.92 puts feet at floor level
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.92

        # Set mocap body to match initial pelvis position (weld constraint keeps it stable)
        if self.model.nmocap > 0:
            self.data.mocap_pos[0] = [0, 0, 0.92]
            self.data.mocap_quat[0] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)  # initialize derived quantities

        # Settle for a short period with first-frame targets to establish contact
        settle_steps = steps_per_frame * 5  # ~5 frames of settling
        first_target = joint_angles[0]
        for _ in range(settle_steps):
            self._apply_pd_control(first_target)
            mujoco.mj_step(self.model, self.data)

        for frame_idx in range(n_frames):
            target = joint_angles[frame_idx]  # (nu,)

            # PD control sub-stepping
            for _ in range(steps_per_frame):
                self._apply_pd_control(target)
                mujoco.mj_step(self.model, self.data)

            # Inverse dynamics for this frame (compute required torques)
            mujoco.mj_inverse(self.model, self.data)

            # Record data
            all_joint_angles.append(self.data.qpos.copy().tolist())
            all_joint_velocities.append(self.data.qvel.copy().tolist())
            all_joint_torques.append(self.data.qfrc_inverse.copy().tolist())

            # Contact forces from touch sensors
            contact_force = self._read_contact_forces()
            all_contact_forces.append(contact_force)

            # Center of mass
            com = self.data.subtree_com[0].copy()  # root body COM
            all_com_positions.append(com.tolist())

        return {
            "joint_angles": all_joint_angles,
            "joint_velocities": all_joint_velocities,
            "joint_torques": all_joint_torques,
            "contact_forces": all_contact_forces,
            "com_position": all_com_positions
        }

    def _apply_pd_control(self, target: np.ndarray):
        """Apply PD control for one physics step."""
        for i in range(self.model.nu):
            q_current = self.data.qpos[self._act_qpos_adr[i]]
            qvel_current = self.data.qvel[self._act_dof_adr[i]]

            error = target[i] - q_current
            torque = self._kp[i] * error - self._kd[i] * qvel_current

            self.data.ctrl[i] = np.clip(torque / self._act_gear[i], -1.0, 1.0)

    def _read_contact_forces(self) -> float:
        """
        Compute total ground reaction force by iterating contacts.

        Sums the normal force magnitude for all contacts involving the
        floor geom (geom ID 0, the ground plane).
        """
        total_force = 0.0
        floor_geom_id = 0  # floor is always geom 0 in our MJCF

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == floor_geom_id or contact.geom2 == floor_geom_id:
                # Extract contact normal force using MuJoCo API
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                # force[0] is the normal force magnitude
                total_force += abs(force[0])

        return total_force

    # ── Dummy Fallback ───────────────────────────────────────────────

    def _generate_dummy_physics_data(self, pose_data_path: Path, output_path: Path) -> Dict:
        """Generate dummy physics data when MuJoCo is not available."""
        print("Generating dummy physics data...")

        with open(pose_data_path, 'r') as f:
            pose_data = json.load(f)

        n_frames = len(pose_data['joints_3d'])
        fps = pose_data.get('fps', 30.0)

        dummy_data = {
            "joint_angles": [[0.0] * 26 for _ in range(n_frames)],
            "joint_velocities": [[0.0] * 26 for _ in range(n_frames)],
            "joint_torques": [[0.0] * 26 for _ in range(n_frames)],
            "contact_forces": [800.0] * n_frames,
            "com_position": [[0.0, 0.0, 1.0] for _ in range(n_frames)],
        }

        with open(output_path, 'w') as f:
            json.dump(dummy_data, f, indent=2)

        return {
            "joint_angles": dummy_data["joint_angles"],
            "joint_velocities": dummy_data["joint_velocities"],
            "joint_torques": dummy_data["joint_torques"],
            "contact_forces": dummy_data["contact_forces"],
            "com_position": dummy_data["com_position"],
            "duration": n_frames / fps if fps > 0 else 0.0,
            "frame_count": n_frames,
            "processing_time": 0.5,
            "model_version": f"dummy_sim_v{self.version.item()}"
        }

    def get_version(self) -> str:
        """Get the simulator version"""
        return f"mujoco_simulator_v{self.version.item()}"
