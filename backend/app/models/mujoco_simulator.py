"""
MuJoCoSimulator - Physics simulation using MuJoCo

This module reconstructs golf swing motion in MuJoCo physics simulator,
computes inverse dynamics, and extracts biomechanics data.
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
    Physics simulation using MuJoCo

    Replays trajectory from pose data in MuJoCo physics environment,
    computes inverse dynamics and extracts biomechanics metrics.
    """

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
                print(f"  - DOF: {self.model.nv}")
                print(f"  - Bodies: {self.model.nbody}")
                print(f"  - Actuators: {self.model.nu}")
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

    def forward(self, pose_data_path: Path, output_path: Path) -> Dict:
        """
        Replay trajectory in MuJoCo and extract physics data

        Args:
            pose_data_path: Path to JSON file with pose keypoints
            output_path: Path to save physics data JSON

        Returns:
            Dictionary with physics data:
                - joint_angles: List of joint angles per frame
                - joint_velocities: List of joint velocities
                - joint_torques: List of computed torques (inverse dynamics)
                - contact_forces: Ground reaction forces
                - com_position: Center of mass trajectory
                - duration: Simulation duration
                - frame_count: Number of simulated frames
                - processing_time: Time taken
                - model_version: Model version
        """
        if not self.mujoco_ready or self.model is None:
            return self._generate_dummy_physics_data(pose_data_path, output_path)

        start_time = time.time()

        # Load pose data
        with open(pose_data_path, 'r') as f:
            pose_data = json.load(f)

        joints_3d = pose_data['joints_3d']  # (T, 17, 3)
        timestamps = pose_data['timestamps']
        fps = pose_data.get('fps', 30.0)

        # Convert pose keypoints to joint angles (simplified mapping)
        joint_angles_sequence = self._map_pose_to_joint_angles(joints_3d)

        # Smooth trajectory to reduce jitter
        joint_angles_sequence = self._smooth_trajectory(joint_angles_sequence)

        # Run simulation
        physics_data = self._simulate_trajectory(joint_angles_sequence, fps)

        # Save physics data to JSON
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

    def _map_pose_to_joint_angles(self, joints_3d: List[List[List[float]]]) -> np.ndarray:
        """
        Map 3D pose keypoints to MuJoCo joint angles

        This is a simplified mapping. In production, would use inverse kinematics.

        Args:
            joints_3d: List of 17 keypoints per frame

        Returns:
            Joint angles array (T, n_joints)
        """
        n_frames = len(joints_3d)
        n_joints = self.model.nv - 6  # Exclude free joint (pelvis 6 DOF)

        # Initialize with neutral pose
        joint_angles = np.zeros((n_frames, n_joints))

        # For now, use a simplified mapping based on keypoint positions
        # This would be replaced with proper IK in production
        for t in range(n_frames):
            keypoints = np.array(joints_3d[t])  # (17, 3)

            # Extract key body part positions
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_elbow = keypoints[7]
            right_elbow = keypoints[8]
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            left_hip = keypoints[11]
            right_hip = keypoints[12]

            # Compute simple joint angles from geometric relationships
            # (This is a placeholder - real implementation would use proper IK)

            # Shoulder flexion (estimated from vertical displacement)
            if t < n_joints:
                joint_angles[t, 0] = (left_wrist[1] - left_shoulder[1]) * 180  # Rough angle

        return joint_angles

    def _smooth_trajectory(self, trajectory: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
        """
        Smooth trajectory using Savitzky-Golay filter

        Args:
            trajectory: Joint angles array (T, n_joints)
            window_length: Filter window length (must be odd)
            polyorder: Polynomial order

        Returns:
            Smoothed trajectory
        """
        if len(trajectory) < window_length:
            return trajectory

        smoothed = np.zeros_like(trajectory)
        for j in range(trajectory.shape[1]):
            smoothed[:, j] = savgol_filter(trajectory[:, j], window_length, polyorder)

        return smoothed

    def _simulate_trajectory(self, joint_angles: np.ndarray, fps: float) -> Dict:
        """
        Run MuJoCo simulation with given joint angles

        Args:
            joint_angles: Target joint angles (T, n_joints)
            fps: Frames per second

        Returns:
            Dictionary with simulation results
        """
        dt = 1.0 / fps
        n_steps = len(joint_angles)

        # Storage for physics data
        all_joint_angles = []
        all_joint_velocities = []
        all_joint_torques = []
        all_contact_forces = []
        all_com_positions = []

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        for step in range(n_steps):
            # Set target joint positions (simplified control)
            # In production, would use mocap or proper PD controller
            self.data.ctrl[:] = 0  # Zero torque for now

            # Step simulation
            mujoco.mj_step(self.model, self.data)

            # Compute inverse dynamics to get required torques
            mujoco.mj_inverse(self.model, self.data)

            # Record data
            all_joint_angles.append(self.data.qpos.copy().tolist())
            all_joint_velocities.append(self.data.qvel.copy().tolist())
            all_joint_torques.append(self.data.qfrc_inverse.copy().tolist())

            # Extract contact forces from constraint forces
            # Use efc_force (constraint forces) instead of contact.force
            contact_force_total = 0.0
            for i in range(min(self.data.ncon, len(self.data.efc_force) // 6)):
                # Each contact can have up to 6 force dimensions
                start_idx = i * 6
                end_idx = start_idx + 6
                if end_idx <= len(self.data.efc_force):
                    force_vec = self.data.efc_force[start_idx:end_idx]
                    contact_force_total += np.linalg.norm(force_vec)
            all_contact_forces.append(float(contact_force_total))

            # Center of mass (use subtree_com for root body)
            # In MuJoCo 3.x, use data.subtree_com which is automatically computed
            if hasattr(self.data, 'subtree_com'):
                com = self.data.subtree_com[0]  # Root body COM
            else:
                # Fallback: compute manually from qpos
                com = self.data.qpos[:3] if len(self.data.qpos) >= 3 else np.zeros(3)
            all_com_positions.append(com.tolist() if isinstance(com, np.ndarray) else [com[0], com[1], com[2]])

        return {
            "joint_angles": all_joint_angles,
            "joint_velocities": all_joint_velocities,
            "joint_torques": all_joint_torques,
            "contact_forces": all_contact_forces,
            "com_position": all_com_positions
        }

    def _generate_dummy_physics_data(self, pose_data_path: Path, output_path: Path) -> Dict:
        """
        Generate dummy physics data when MuJoCo is not available
        """
        print("Generating dummy physics data...")

        # Load pose data to get frame count
        with open(pose_data_path, 'r') as f:
            pose_data = json.load(f)

        n_frames = len(pose_data['joints_3d'])
        fps = pose_data.get('fps', 30.0)

        # Generate dummy data
        dummy_data = {
            "joint_angles": [[0.0] * 26 for _ in range(n_frames)],
            "joint_velocities": [[0.0] * 26 for _ in range(n_frames)],
            "joint_torques": [[0.0] * 26 for _ in range(n_frames)],
            "contact_forces": [800.0] * n_frames,  # Typical body weight force
            "com_position": [[0.0, 0.0, 1.0] for _ in range(n_frames)],
        }

        # Save to output
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
