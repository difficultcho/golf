"""
PhysicsAnalyzer - Biomechanics metrics computation

This module analyzes physics data from MuJoCo simulation and computes
golf swing biomechanics metrics like club speed, X-Factor, energy efficiency, etc.
"""

import torch
import torch.nn as nn
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List


class PhysicsAnalyzer(nn.Module):
    """
    Compute biomechanics metrics from physics simulation data

    Analyzes joint torques, velocities, and contact forces to extract
    meaningful golf swing performance indicators.
    """

    def __init__(self):
        super().__init__()
        self.version = nn.Parameter(torch.tensor([2.0]), requires_grad=False)

    def forward(self, physics_data_path: Path, output_path: Path) -> Dict:
        """
        Analyze physics data and compute swing metrics

        Args:
            physics_data_path: Path to JSON with MuJoCo simulation data
            output_path: Path to save metrics JSON

        Returns:
            Dictionary with metrics:
                - club_head_speed_mph: Peak club head speed (mph)
                - swing_duration_sec: Total swing duration (seconds)
                - peak_torques: Peak torques for major joints (Nm)
                - energy_efficiency: Energy transfer efficiency (0-1)
                - balance_score: Balance/stability score (0-100)
                - x_factor: Shoulder-hip separation angle (degrees)
        """
        start_time = time.time()

        # Load physics data
        with open(physics_data_path, 'r') as f:
            physics_data = json.load(f)

        joint_angles = np.array(physics_data['joint_angles'])  # (T, n_joints)
        joint_velocities = np.array(physics_data['joint_velocities'])
        joint_torques = np.array(physics_data['joint_torques'])
        contact_forces = np.array(physics_data['contact_forces'])
        com_positions = np.array(physics_data['com_position'])

        n_frames = len(joint_angles)

        # Compute metrics
        metrics = {}

        # 1. Club Head Speed (estimate from wrist velocity)
        metrics['club_head_speed_mph'] = self._compute_club_head_speed(joint_velocities)

        # 2. Swing Duration
        metrics['swing_duration_sec'] = float(n_frames / 30.0)  # Assuming 30fps

        # 3. Peak Torques
        metrics['peak_torques'] = self._compute_peak_torques(joint_torques)

        # 4. Energy Efficiency
        metrics['energy_efficiency'] = self._compute_energy_efficiency(
            joint_velocities, joint_torques
        )

        # 5. Balance Score
        metrics['balance_score'] = self._compute_balance_score(
            com_positions, contact_forces
        )

        # 6. X-Factor (shoulder-hip separation)
        metrics['x_factor'] = self._compute_x_factor(joint_angles)

        # Save metrics
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        processing_time = time.time() - start_time

        return {
            **metrics,
            "processing_time": processing_time,
            "model_version": f"physics_analyzer_v{self.version.item()}"
        }

    def _compute_club_head_speed(self, joint_velocities: np.ndarray) -> float:
        """
        Estimate club head speed from joint velocities

        In a full implementation, would track club head position/velocity directly.
        Here we estimate from wrist velocity as a proxy.

        Args:
            joint_velocities: Joint velocities array (T, n_joints)

        Returns:
            Peak club head speed in mph
        """
        if len(joint_velocities) == 0:
            return 0.0

        # Estimate from wrist velocity (simplified)
        # Assume wrist velocity is proportional to club head speed
        # Typical club length: 1.15m, adds ~2x multiplier to wrist speed

        # Use velocity magnitude across all joints as proxy
        velocity_magnitudes = np.linalg.norm(joint_velocities, axis=1)
        peak_velocity_mps = np.max(velocity_magnitudes)

        # Convert m/s to mph and apply club length multiplier
        club_head_speed_mph = peak_velocity_mps * 2.23694 * 2.0  # m/s to mph, with club multiplier

        # Clamp to reasonable range for golf (60-130 mph)
        club_head_speed_mph = np.clip(club_head_speed_mph, 60.0, 130.0)

        return float(club_head_speed_mph)

    def _compute_peak_torques(self, joint_torques: np.ndarray) -> Dict[str, float]:
        """
        Compute peak torques for major joints

        Args:
            joint_torques: Joint torques array (T, n_joints)

        Returns:
            Dictionary of peak torques (Nm)
        """
        if len(joint_torques) == 0:
            return {"shoulder": 0.0, "hip": 0.0, "knee": 0.0}

        # Compute RMS torque for each joint
        rms_torques = np.sqrt(np.mean(joint_torques ** 2, axis=0))

        # Extract representative joints (simplified indexing)
        # In full implementation, would map to specific joint names
        peak_torques = {
            "shoulder": float(rms_torques[0]) if len(rms_torques) > 0 else 0.0,
            "hip": float(rms_torques[1]) if len(rms_torques) > 1 else 0.0,
            "knee": float(rms_torques[2]) if len(rms_torques) > 2 else 0.0,
        }

        return peak_torques

    def _compute_energy_efficiency(
        self, joint_velocities: np.ndarray, joint_torques: np.ndarray
    ) -> float:
        """
        Compute energy transfer efficiency

        Measures how efficiently energy flows from legs through torso to arms/club.

        Args:
            joint_velocities: Velocities (T, n_joints)
            joint_torques: Torques (T, n_joints)

        Returns:
            Efficiency score (0-1)
        """
        if len(joint_velocities) == 0 or len(joint_torques) == 0:
            return 0.5

        # Compute mechanical power: P = τ * ω
        power = np.abs(joint_torques * joint_velocities)

        # Total power delivered
        total_power = np.sum(power)

        # Power in distal segments (arms/wrists) - represents useful output
        # Simplified: assume last 30% of joints are distal
        n_joints = joint_velocities.shape[1]
        distal_start = int(n_joints * 0.7)
        distal_power = np.sum(power[:, distal_start:])

        # Efficiency = useful output / total input
        efficiency = distal_power / (total_power + 1e-6)  # Avoid division by zero

        # Clamp to [0, 1]
        efficiency = np.clip(efficiency, 0.0, 1.0)

        return float(efficiency)

    def _compute_balance_score(
        self, com_positions: np.ndarray, contact_forces: np.ndarray
    ) -> float:
        """
        Compute balance and stability score

        Analyzes center of mass stability and ground contact consistency.

        Args:
            com_positions: COM positions (T, 3)
            contact_forces: Contact forces (T,)

        Returns:
            Balance score (0-100)
        """
        if len(com_positions) == 0:
            return 50.0

        # 1. COM stability: lower variance = better balance
        com_variance = np.var(com_positions, axis=0)
        stability_score = 1.0 / (1.0 + np.sum(com_variance))

        # 2. Ground contact consistency: stable contact = better balance
        if len(contact_forces) > 0:
            contact_variance = np.var(contact_forces)
            contact_score = 1.0 / (1.0 + contact_variance / 1000.0)
        else:
            contact_score = 0.5

        # Combine scores
        balance_score = (stability_score + contact_score) / 2.0

        # Scale to 0-100
        balance_score = balance_score * 100.0

        return float(np.clip(balance_score, 0.0, 100.0))

    def _compute_x_factor(self, joint_angles: np.ndarray) -> float:
        """
        Compute X-Factor (shoulder-hip separation angle)

        Key metric in golf biomechanics. Measures rotational separation
        between shoulders and hips during backswing.

        Args:
            joint_angles: Joint angles (T, n_joints)

        Returns:
            X-Factor angle in degrees
        """
        if len(joint_angles) == 0:
            return 0.0

        # In full implementation, would extract specific shoulder/hip rotation joints
        # Here we estimate from joint angle variance

        # Find the frame with maximum shoulder-hip difference
        # Simplified: use variance in first few joints as proxy
        if joint_angles.shape[1] >= 2:
            shoulder_angle = joint_angles[:, 0]  # Approximate shoulder rotation
            hip_angle = joint_angles[:, 1]  # Approximate hip rotation

            # X-Factor is max separation during backswing (first 40% of swing)
            backswing_end = int(len(joint_angles) * 0.4)
            if backswing_end > 0:
                separation = np.abs(shoulder_angle[:backswing_end] - hip_angle[:backswing_end])
                x_factor = np.max(separation) * 180.0 / np.pi  # Convert to degrees
            else:
                x_factor = 0.0
        else:
            x_factor = 0.0

        # Typical X-Factor range: 30-60 degrees
        x_factor = np.clip(x_factor, 0.0, 90.0)

        return float(x_factor)

    def get_version(self) -> str:
        """Get the analyzer version"""
        return f"physics_analyzer_v{self.version.item()}"
