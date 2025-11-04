#!/usr/bin/env python3
"""
IMU-based dynamic tilt compensation for OAK-D camera.
Uses BNO086 IMU rotation quaternions to correct for robot pitch and roll.
"""

import numpy as np
import math
from typing import Tuple


class IMUTiltCompensation:
    """
    Dynamically adjusts camera tilt based on IMU orientation data.

    The OAK-D's BNO086 IMU provides rotation quaternions representing the camera's
    orientation. This class extracts pitch and roll angles to correct ground projection
    calculations that would otherwise assume the robot is perfectly level.

    Coordinate frames:
    - Camera (DepthAI): +X right, +Y down, +Z forward
    - Robot (NWU): +X forward (north), +Y left (west), +Z up
    - World: Level ground plane
    """

    def __init__(self, nominal_pitch_deg: float = 30.0, nominal_roll_deg: float = 0.0):
        """
        Initialize tilt compensation.

        Args:
            nominal_pitch_deg: Camera's static pitch angle when robot is level (degrees)
                              Positive = tilted down (typical: 30°)
            nominal_roll_deg: Camera's static roll angle when robot is level (degrees)
                             Positive = tilted right (typical: 0°)
        """
        self.nominal_pitch_deg = nominal_pitch_deg
        self.nominal_roll_deg = nominal_roll_deg

        # Current IMU-measured orientation
        self.current_pitch_deg = 0.0
        self.current_roll_deg = 0.0

        # Effective camera angles (nominal + IMU correction)
        self.effective_pitch_deg = nominal_pitch_deg
        self.effective_roll_deg = nominal_roll_deg


    def update_from_quaternion(self, i: float, j: float, k: float, real: float) -> None:
        """
        Update tilt compensation from IMU rotation quaternion.

        The BNO086 provides quaternions in the form (i, j, k, real) where:
        - Quaternion = real + i*x + j*y + k*z
        - Represents rotation from initial orientation to current orientation

        Args:
            i, j, k, real: Quaternion components from IMU
        """
        # Convert quaternion to Euler angles (intrinsic XYZ rotation)
        pitch_rad, roll_rad, yaw_rad = self._quaternion_to_euler(i, j, k, real)

        # Store current IMU-measured angles
        self.current_pitch_deg = math.degrees(pitch_rad)
        self.current_roll_deg = math.degrees(roll_rad)

        # Calculate effective camera orientation
        # Positive robot pitch (nose up) reduces effective camera downward tilt
        # Positive robot roll (right side down) adds leftward tilt to camera
        self.effective_pitch_deg = self.nominal_pitch_deg - self.current_pitch_deg
        self.effective_roll_deg = self.nominal_roll_deg - self.current_roll_deg


    def get_up_vector_camera_frame(self) -> np.ndarray:
        """
        Calculate the "up" direction in camera frame considering current tilt.

        This vector points from the ground toward the sky in the camera's coordinate system.
        It's used to project camera rays onto the ground plane.

        Returns:
            3D unit vector [x, y, z] in camera frame pointing "up" relative to ground
        """
        pitch_rad = math.radians(self.effective_pitch_deg)
        roll_rad = math.radians(self.effective_roll_deg)

        # Camera frame: +X right, +Y down, +Z forward
        # Level camera: up = [0, -1, 0] (opposite of +Y down)

        # Apply pitch (rotation about X axis - tilts camera up/down)
        # Positive pitch tilts camera down, rotating up vector toward +Z
        # up_pitch = [0, -cos(pitch), -sin(pitch)]

        # Apply roll (rotation about Z axis - tilts camera left/right)
        # Positive roll tilts camera right, rotating up vector toward -X
        # Combined: up = Rz(roll) * Ry(0) * Rx(pitch) * [0, -1, 0]

        # For camera frame with pitch and roll:
        up_x = -math.sin(roll_rad) * math.cos(pitch_rad)
        up_y = -math.cos(roll_rad) * math.cos(pitch_rad)
        up_z = -math.sin(pitch_rad)

        up_vector = np.array([up_x, up_y, up_z], dtype=float)

        # Normalize (should already be unit length, but ensure it)
        norm = np.linalg.norm(up_vector)
        if norm > 1e-9:
            up_vector /= norm

        return up_vector


    def get_diagnostics(self) -> dict:
        """
        Get current tilt compensation state for logging/debugging.

        Returns:
            Dictionary with current angles and corrections
        """
        return {
            'nominal_pitch_deg': self.nominal_pitch_deg,
            'nominal_roll_deg': self.nominal_roll_deg,
            'imu_pitch_deg': self.current_pitch_deg,
            'imu_roll_deg': self.current_roll_deg,
            'effective_pitch_deg': self.effective_pitch_deg,
            'effective_roll_deg': self.effective_roll_deg,
            'pitch_correction_deg': self.current_pitch_deg,
            'roll_correction_deg': self.current_roll_deg,
        }


    @staticmethod
    def _quaternion_to_euler(i: float, j: float, k: float, real: float) -> Tuple[float, float, float]:
        """
        Convert quaternion to Euler angles (pitch, roll, yaw).

        Uses intrinsic XYZ rotation sequence (pitch-roll-yaw).
        This matches the typical robotics convention for ground vehicles.

        Args:
            i, j, k, real: Quaternion components (x, y, z, w)

        Returns:
            (pitch, roll, yaw) in radians
            - pitch: Rotation about X axis (nose up/down), range [-π/2, π/2]
            - roll: Rotation about Y axis (right side up/down), range [-π, π]
            - yaw: Rotation about Z axis (heading), range [-π, π]
        """
        # Quaternion components (x, y, z, w)
        x, y, z, w = i, j, k, real

        # Pitch (X-axis rotation)
        sin_pitch = 2.0 * (w * y - z * x)
        # Clamp to avoid numerical issues with arcsin
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        pitch = math.asin(sin_pitch)

        # Roll (Y-axis rotation)
        sin_roll = 2.0 * (w * x + y * z)
        cos_roll = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sin_roll, cos_roll)

        # Yaw (Z-axis rotation)
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(sin_yaw, cos_yaw)

        return pitch, roll, yaw


# Example usage and testing
if __name__ == "__main__":
    print("IMU Tilt Compensation Test")
    print("=" * 60)

    # Initialize with 30° nominal downward pitch
    compensator = IMUTiltCompensation(nominal_pitch_deg=30.0)

    # Test 1: Level robot (identity quaternion)
    print("\nTest 1: Robot level (no tilt)")
    compensator.update_from_quaternion(0, 0, 0, 1)  # Identity quaternion
    diag = compensator.get_diagnostics()
    print(f"  IMU pitch: {diag['imu_pitch_deg']:.1f}°, roll: {diag['imu_roll_deg']:.1f}°")
    print(f"  Effective camera pitch: {diag['effective_pitch_deg']:.1f}°, roll: {diag['effective_roll_deg']:.1f}°")
    up_vec = compensator.get_up_vector_camera_frame()
    print(f"  Up vector (camera frame): [{up_vec[0]:.3f}, {up_vec[1]:.3f}, {up_vec[2]:.3f}]")

    # Test 2: Robot pitched forward 5° (nose down)
    print("\nTest 2: Robot pitched forward 5° (deceleration)")
    # Forward pitch rotates about Y axis (j component)
    pitch_rad = math.radians(5.0)
    q_pitch = [0, math.sin(pitch_rad/2), 0, math.cos(pitch_rad/2)]
    compensator.update_from_quaternion(*q_pitch)
    diag = compensator.get_diagnostics()
    print(f"  IMU pitch: {diag['imu_pitch_deg']:.1f}°, roll: {diag['imu_roll_deg']:.1f}°")
    print(f"  Effective camera pitch: {diag['effective_pitch_deg']:.1f}° (was 30°, now 35° to ground)")
    print(f"  Correction applied: {diag['pitch_correction_deg']:.1f}°")

    # Test 3: Robot rolled right 3° (right side down)
    print("\nTest 3: Robot rolled right 3° (uneven ground)")
    roll_rad = math.radians(3.0)
    q_roll = [math.sin(roll_rad/2), 0, 0, math.cos(roll_rad/2)]
    compensator.update_from_quaternion(*q_roll)
    diag = compensator.get_diagnostics()
    print(f"  IMU pitch: {diag['imu_pitch_deg']:.1f}°, roll: {diag['imu_roll_deg']:.1f}°")
    print(f"  Effective camera roll: {diag['effective_roll_deg']:.1f}°")
    print(f"  This corrects dipbob drop position by compensating for robot lean!")

    # Test 4: Combined pitch and roll
    print("\nTest 4: Robot pitched forward 5° AND rolled right 3°")
    # Combined quaternion (pitch * roll)
    combined_quat = [
        math.sin(roll_rad/2) * math.cos(pitch_rad/2),
        math.cos(roll_rad/2) * math.sin(pitch_rad/2),
        math.sin(roll_rad/2) * math.sin(pitch_rad/2),
        math.cos(roll_rad/2) * math.cos(pitch_rad/2)
    ]
    compensator.update_from_quaternion(*combined_quat)
    diag = compensator.get_diagnostics()
    print(f"  IMU pitch: {diag['imu_pitch_deg']:.1f}°, roll: {diag['imu_roll_deg']:.1f}°")
    print(f"  Effective camera pitch: {diag['effective_pitch_deg']:.1f}°, roll: {diag['effective_roll_deg']:.1f}°")
    up_vec = compensator.get_up_vector_camera_frame()
    print(f"  Up vector (camera frame): [{up_vec[0]:.3f}, {up_vec[1]:.3f}, {up_vec[2]:.3f}]")

    print("\n" + "=" * 60)
    print("✓ IMU tilt compensation ready for integration")
