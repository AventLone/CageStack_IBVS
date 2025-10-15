"""
SE2 (Special Euclidean Group in 2D) utility class for coordinate transformations.

This module provides utilities for working with 2D rigid body transformations,
including rotation and translation operations in the SE(2) group.
"""

import numpy as np
from typing import Union, Tuple
from .angle import angle_mod


class SE2:
    """
    SE(2) transformation class for 2D coordinate transformations.

    Represents a 2D rigid body transformation consisting of rotation and translation.
    The transformation can be represented as a 3x3 homogeneous transformation matrix:

    T = | R  t |
        | 0  1 |

    where R is a 2x2 rotation matrix and t is a 2x1 translation vector.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """
        Initialize SE2 transformation.

        Args:
            x (float): Translation in x direction
            y (float): Translation in y direction
            theta (float): Rotation angle in radians
        """
        self.x = x
        self.y = y
        self.theta = angle_mod(theta)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "SE2":
        """
        Create SE2 transformation from 3x3 homogeneous transformation matrix.

        Args:
            matrix (np.ndarray): 3x3 transformation matrix

        Returns:
            SE2: SE2 transformation object
        """
        if matrix.shape != (3, 3):
            raise ValueError("Matrix must be 3x3")

        x = float(matrix[0, 2])
        y = float(matrix[1, 2])
        theta = float(np.arctan2(matrix[1, 0], matrix[0, 0]))

        return cls(x, y, theta)

    @classmethod
    def from_pose(cls, pose: Union[np.ndarray, list, tuple]) -> "SE2":
        """
        Create SE2 transformation from pose vector [x, y, theta].

        Args:
            pose: Pose as [x, y, theta] array, list, or tuple

        Returns:
            SE2: SE2 transformation object
        """
        pose = np.asarray(pose)
        if pose.shape != (3,):
            raise ValueError("Pose must have shape (3,)")

        return cls(pose[0], pose[1], pose[2])

    def to_matrix(self) -> np.ndarray:
        """
        Convert to 3x3 homogeneous transformation matrix.

        Returns:
            np.ndarray: 3x3 transformation matrix
        """
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        return np.array([[cos_theta, -sin_theta, self.x], [sin_theta, cos_theta, self.y], [0, 0, 1]])

    def to_pose(self) -> np.ndarray:
        """
        Convert to pose vector [x, y, theta].

        Returns:
            np.ndarray: Pose as [x, y, theta]
        """
        return np.array([self.x, self.y, self.theta])

    def rotation_matrix(self) -> np.ndarray:
        """
        Get 2x2 rotation matrix.

        Returns:
            np.ndarray: 2x2 rotation matrix
        """
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    def translation_vector(self) -> np.ndarray:
        """
        Get translation vector.

        Returns:
            np.ndarray: Translation vector [x, y]
        """
        return np.array([self.x, self.y])

    def transform_point(self, point: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """
        Transform a 2D point using this SE2 transformation.

        Args:
            point: 2D point as [x, y] array, list, or tuple

        Returns:
            np.ndarray: Transformed point [x, y]
        """
        point = np.asarray(point)
        if point.shape == (2,):
            # Convert to homogeneous coordinates
            point_h = np.array([point[0], point[1], 1])
        elif point.shape == (3,) and point[2] == 1:
            point_h = point
        else:
            raise ValueError("Point must be 2D [x, y] or homogeneous [x, y, 1]")

        transformed = self.to_matrix() @ point_h
        return transformed[:2]

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform multiple 2D points using this SE2 transformation.

        Args:
            points (np.ndarray): Array of points with shape (N, 2)

        Returns:
            np.ndarray: Transformed points with shape (N, 2)
        """
        points = np.asarray(points)
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError("Points must have shape (N, 2)")

        # Convert to homogeneous coordinates
        n_points = points.shape[0]
        points_h = np.ones((n_points, 3))
        points_h[:, :2] = points

        # Apply transformation
        T = self.to_matrix()
        transformed_h = (T @ points_h.T).T

        return transformed_h[:, :2]

    def inverse(self) -> "SE2":
        """
        Compute inverse transformation.

        Returns:
            SE2: Inverse SE2 transformation
        """
        # For SE(2), inverse is:
        # T^-1 = | R^T  -R^T*t |
        #        | 0    1      |
        R = self.rotation_matrix()
        t = self.translation_vector()

        R_inv = R.T
        t_inv = -R_inv @ t
        theta_inv = float(angle_mod(-self.theta))

        return SE2(float(t_inv[0]), float(t_inv[1]), theta_inv)

    def compose(self, other: "SE2") -> "SE2":
        """
        Compose this transformation with another SE2 transformation.

        Args:
            other (SE2): Other SE2 transformation

        Returns:
            SE2: Composed transformation (self * other)
        """
        T1 = self.to_matrix()
        T2 = other.to_matrix()
        T_composed = T1 @ T2

        return SE2.from_matrix(T_composed)

    def relative_to(self, other: "SE2") -> "SE2":
        """
        Compute relative transformation from other to self.

        Args:
            other (SE2): Reference SE2 transformation

        Returns:
            SE2: Relative transformation (other^-1 * self)
        """
        return other.inverse().compose(self)

    def distance_to(self, other: "SE2") -> float:
        """
        Compute Euclidean distance between translation components.

        Args:
            other (SE2): Other SE2 transformation

        Returns:
            float: Euclidean distance
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx * dx + dy * dy)

    def angle_difference_to(self, other: "SE2") -> float:
        """
        Compute angle difference to another SE2 transformation.

        Args:
            other (SE2): Other SE2 transformation

        Returns:
            float: Angle difference in radians [-pi, pi]
        """
        return float(angle_mod(other.theta - self.theta))

    def __mul__(self, other: "SE2") -> "SE2":
        """
        Compose transformations using * operator.

        Args:
            other (SE2): Other SE2 transformation

        Returns:
            SE2: Composed transformation
        """
        return self.compose(other)

    def __repr__(self) -> str:
        """String representation of SE2 transformation."""
        return f"SE2(x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f})"

    def __str__(self) -> str:
        """String representation of SE2 transformation."""
        return self.__repr__()


def transform_points_batch(transformations: list, points: np.ndarray) -> np.ndarray:
    """
    Transform points using a batch of SE2 transformations.

    Args:
        transformations (list): List of SE2 transformations
        points (np.ndarray): Array of points with shape (N, 2)

    Returns:
        np.ndarray: Transformed points with shape (len(transformations), N, 2)
    """
    points = np.asarray(points)
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2)")

    results = []
    for transform in transformations:
        results.append(transform.transform_points(points))

    return np.array(results)


def interpolate_se2(se2_start: SE2, se2_end: SE2, t: float) -> SE2:
    """
    Interpolate between two SE2 transformations.

    Args:
        se2_start (SE2): Starting SE2 transformation
        se2_end (SE2): Ending SE2 transformation
        t (float): Interpolation parameter [0, 1]

    Returns:
        SE2: Interpolated SE2 transformation
    """
    if not (0 <= t <= 1):
        raise ValueError("Interpolation parameter t must be in [0, 1]")

    # Linear interpolation for translation
    x = se2_start.x + t * (se2_end.x - se2_start.x)
    y = se2_start.y + t * (se2_end.y - se2_start.y)

    # SLERP for rotation (shortest path)
    theta_diff = float(angle_mod(se2_end.theta - se2_start.theta))
    theta = se2_start.theta + t * theta_diff

    return SE2(x, y, float(theta))


def create_se2_from_points(point1: Union[np.ndarray, list, tuple], point2: Union[np.ndarray, list, tuple]) -> SE2:
    """
    Create SE2 transformation from two points.
    The transformation will be positioned at point1 and oriented towards point2.

    Args:
        point1: Starting point [x, y]
        point2: Target point [x, y]

    Returns:
        SE2: SE2 transformation
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)

    if point1.shape != (2,) or point2.shape != (2,):
        raise ValueError("Points must be 2D [x, y]")

    x, y = float(point1[0]), float(point1[1])
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    theta = float(np.arctan2(dy, dx))

    return SE2(x, y, theta)
