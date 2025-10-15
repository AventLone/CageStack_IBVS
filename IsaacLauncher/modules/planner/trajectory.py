"""
Trajectory class for path tracking with interpolation and projection capabilities.

This module provides a Trajectory class that can:
- Store waypoints with coordinates, angles, and directions
- Perform interpolation between waypoints
- Find nearest projection points for given poses
- Calculate longitudinal and lateral coordinates relative to the path

Author: Assistant
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional, TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from loguru import logger

# Import configuration management
if TYPE_CHECKING:
    from .config import TrajectoryConfig


@dataclass
class Waypoint:
    """Represents a point on the trajectory (can be waypoint or interpolated point)."""

    x: float
    y: float
    yaw: float
    direction: int = 1


@dataclass
class ProjectedPoint:
    """Represents a point projected onto the trajectory."""

    x: float
    y: float
    yaw: float
    s: float  # distance along path
    direction: int = 1


@dataclass
class FrenetCoordinates:
    """Represents Frenet coordinates relative to the trajectory."""

    s: float  # longitudinal distance along path
    d: float  # lateral distance from path (positive = left)


class Trajectory:
    """
    A trajectory class that supports interpolation and projection operations for one-directional paths.

    The trajectory consists of waypoints with coordinates and heading angles, moving in a single forward direction.
    It provides methods for interpolation and finding projection points along the path.
    """

    def __init__(
        self,
        config: "TrajectoryConfig",
        waypoints: List[Waypoint] = [],
    ):
        """
        Initialize the trajectory.

        Args:
            config: Configuration object.
            waypoints: List of Waypoint objects defining the trajectory
        """
        self.config = config
        self._discretization_distance = self.config.discretization_distance
        self._sample_count = self.config.default_sample_count

        self.waypoints = waypoints.copy()  # Make a copy to avoid modifying the input list
        self._s_values: List[float] = []
        self._interpolators: Dict[str, interp1d] = {}

        # For discrete point sampling
        self._sampled_s: List[float] = []
        self._sampled_waypoints: List[Waypoint] = []

        self._update_path_parameters()

    def add_waypoint(self, x: float, y: float, yaw: float, direction: int = 1):
        """
        Add a waypoint to the trajectory.

        Args:
            x: X coordinate
            y: Y coordinate
            yaw: Heading angle in radians
            direction: Movement direction (1 for forward, -1 for backward)
        """
        waypoint = Waypoint(x, y, yaw, direction)
        self.waypoints.append(waypoint)
        self._update_path_parameters()

    def add_waypoints_from_arrays(
        self,
        x_coords: List[float],
        y_coords: List[float],
        yaw_angles: List[float],
        directions: List[int] = [],
    ):
        """
        Add multiple waypoints from coordinate arrays.

        Args:
            x_coords: List of X coordinates
            y_coords: List of Y coordinates
            yaw_angles: List of heading angles in radians
            directions: List of movement directions (default: all forward)
        """
        if not directions:  # If directions is empty
            directions = [1] * len(x_coords)

        if not (len(x_coords) == len(y_coords) == len(yaw_angles) == len(directions)):
            raise ValueError("All input arrays must have the same length")

        for x, y, yaw, direction in zip(x_coords, y_coords, yaw_angles, directions):
            self.add_waypoint(x, y, yaw, direction)

    def _update_path_parameters(self):
        """Update internal path parameters after waypoint changes."""
        if len(self.waypoints) < 2:
            self._s_values = []
            self._interpolators = {}
            self._sampled_s = []
            self._sampled_waypoints = []
            return

        # Calculate cumulative distances along the path
        self._s_values = [0.0]
        for i in range(1, len(self.waypoints)):
            prev_wp = self.waypoints[i - 1]
            curr_wp = self.waypoints[i]
            distance = math.sqrt((curr_wp.x - prev_wp.x) ** 2 + (curr_wp.y - prev_wp.y) ** 2)
            self._s_values.append(self._s_values[-1] + distance)

        # Create interpolation functions
        if len(self.waypoints) >= 2:
            x_coords = [wp.x for wp in self.waypoints]
            y_coords = [wp.y for wp in self.waypoints]
            yaw_angles = [wp.yaw for wp in self.waypoints]
            directions = [wp.direction for wp in self.waypoints]

            # Handle angle wrapping for yaw interpolation
            yaw_unwrapped = self._unwrap_angles(yaw_angles)

            # Use 'extrapolate' as fill_value to allow extrapolation outside the path range
            self._interpolators = {
                "x": interp1d(
                    self._s_values,
                    x_coords,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",  # Allow extrapolation
                ),
                "y": interp1d(
                    self._s_values,
                    y_coords,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",  # Allow extrapolation
                ),
                "yaw": interp1d(
                    self._s_values,
                    yaw_unwrapped,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",  # Allow extrapolation
                ),
                "direction": interp1d(
                    self._s_values,
                    directions,
                    kind="nearest",
                    bounds_error=False,
                    fill_value="extrapolate",  # Allow extrapolation
                ),
            }

            # Calculate number of samples based on path length and discretization distance
            path_length = self._s_values[-1]
            self._sample_count = max(1, int(path_length / self._discretization_distance))

            # Create discrete samples for nearest point search
            self._sampled_s = list(np.linspace(0, self._s_values[-1], self._sample_count))

            # Pre-compute sampled points
            self._sampled_waypoints = []
            for s in self._sampled_s:
                point = self.interpolate_at_distance(s)
                self._sampled_waypoints.append(point)

    def _unwrap_angles(self, angles: List[float]) -> List[float]:
        """Unwrap angles to avoid discontinuities in interpolation."""
        if len(angles) <= 1:
            return angles

        unwrapped = [angles[0]]
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i - 1]
            # Wrap difference to [-pi, pi]
            while diff > math.pi:
                diff -= 2 * math.pi
            while diff < -math.pi:
                diff += 2 * math.pi
            unwrapped.append(unwrapped[-1] + diff)

        return unwrapped

    def interpolate_at_distance(self, s: float) -> Waypoint:
        """
        Interpolate trajectory at a given cumulative distance.

        Args:
            s: Cumulative distance along the trajectory

        Returns:
            Waypoint containing interpolated position, heading and direction
        """
        if not self._interpolators:
            raise ValueError("Trajectory must have at least 2 waypoints for interpolation")

        x = float(self._interpolators["x"](s))
        y = float(self._interpolators["y"](s))
        yaw = float(self._interpolators["yaw"](s))
        direction = int(self._interpolators["direction"](s))

        # Wrap yaw angle to [-pi, pi]
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        return Waypoint(x=x, y=y, yaw=yaw, direction=direction)

    def find_nearest_point_around(
        self,
        target_x: float,
        target_y: float,
        initial_s: float,
        search_range_meters: float = 10.0,
        tolerance: float = 0.001,
    ) -> float:
        """
        Find the nearest point on trajectory around a given initial position.

        Args:
            target_x: X coordinate of target point
            target_y: Y coordinate of target point
            initial_s: Initial position along trajectory to search around
            search_range_meters: Search range in meters around initial position (default: 10.0)
            tolerance: Search tolerance (default: 0.001)

        Returns:
            Distance along trajectory (s) to the nearest point
        """
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2

        # Define search bounds around initial value
        half_range = search_range_meters / 2
        left = initial_s - half_range
        right = initial_s + half_range

        # Get total path length for clamping
        total_length = self.get_trajectory_length()

        def distance_squared(s):
            s = max(0, min(s, total_length))  # Clamp to valid range
            point = self.interpolate_at_distance(s)
            return (point.x - target_x) ** 2 + (point.y - target_y) ** 2

        # Golden section search
        while (right - left) > tolerance:
            s1 = right - (right - left) / phi
            s2 = left + (right - left) / phi

            if distance_squared(s1) > distance_squared(s2):
                left = s1
            else:
                right = s2

        return (left + right) / 2

    def find_nearest_discrete_point(self, pose_x: float, pose_y: float) -> Tuple[float, int]:
        """
        Find the nearest point among discretely sampled points on trajectory.
        This is the first step in finding the exact nearest point.

        Args:
            pose_x: X coordinate of the query pose
            pose_y: Y coordinate of the query pose

        Returns:
            Tuple of (distance along trajectory, index of nearest sample)

        Raises:
            ValueError: If trajectory is not properly initialized
        """
        if not self._interpolators or not self._sampled_waypoints or not self._sampled_s:
            raise ValueError("Trajectory must have at least 2 waypoints")

        # Convert waypoints to numpy array for efficient distance calculation
        points = np.array([[wp.x, wp.y] for wp in self._sampled_waypoints])
        query_point = np.array([[pose_x, pose_y]])
        distances = cdist(query_point, points)[0]
        nearest_idx = int(np.argmin(distances))
        nearest_s = float(self._sampled_s[nearest_idx])

        return nearest_s, nearest_idx

    def find_nearest_point(self, pose_x: float, pose_y: float) -> ProjectedPoint:
        """
        Find the nearest point on the trajectory to a given pose.
        This is a two-step process:
        1. Find the nearest discretely sampled point
        2. Refine the search around that point to find the exact nearest point on the curve

        Args:
            pose_x: X coordinate of the query pose
            pose_y: Y coordinate of the query pose

        Returns:
            ProjectedPoint containing nearest point data and path distance

        Raises:
            ValueError: If trajectory is empty or not properly initialized
        """
        if not self._interpolators:
            raise ValueError("Trajectory must have at least 2 waypoints")

        # Get trajectory bounds
        total_length = self.get_trajectory_length()

        # Step 1: Find nearest discrete point
        try:
            nearest_s, _ = self.find_nearest_discrete_point(pose_x, pose_y)
        except ValueError:
            # If point is far outside trajectory, use endpoint based on x-coordinate
            if pose_x < self.waypoints[0].x:
                nearest_s = 0.0
            else:
                nearest_s = total_length

        # Step 2: Refine the search around the initial guess
        refined_s = self.find_nearest_point_around(
            target_x=pose_x,
            target_y=pose_y,
            initial_s=nearest_s,
            search_range_meters=10.0,  # Search within 10 meters
            tolerance=0.001,
        )

        # Ensure s is within valid bounds
        refined_s = max(0.0, min(refined_s, total_length))

        point = self.interpolate_at_distance(refined_s)
        return ProjectedPoint(x=point.x, y=point.y, yaw=point.yaw, direction=point.direction, s=refined_s)

    def get_frenet_coordinates(self, pose_x: float, pose_y: float) -> FrenetCoordinates:
        """
        Get Frenet coordinates (longitudinal and lateral) for a given pose.

        Args:
            pose_x: X coordinate of the query pose
            pose_y: Y coordinate of the query pose

        Returns:
            FrenetCoordinates containing longitudinal (s) and lateral (d) distances

        Raises:
            ValueError: If trajectory is empty or not properly initialized
        """
        if not self._interpolators:
            raise ValueError("Trajectory must have at least 2 waypoints")

        try:
            nearest = self.find_nearest_point(pose_x, pose_y)

            # Calculate lateral distance (signed)
            # Vector from nearest point to query pose
            dx = pose_x - nearest.x
            dy = pose_y - nearest.y

            # Normal vector to the trajectory (pointing left)
            normal_x = -math.sin(nearest.yaw)
            normal_y = math.cos(nearest.yaw)

            # Project displacement onto normal vector
            lateral_distance = dx * normal_x + dy * normal_y

            return FrenetCoordinates(s=nearest.s, d=lateral_distance)

        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to get Frenet coordinates: {e}")

    def get_trajectory_length(self) -> float:
        """Get the total length of the trajectory."""
        if self._s_values is None:
            return 0.0
        return self._s_values[-1]

    def get_waypoint_distances(self) -> List[float]:
        """Get the cumulative distances to each waypoint."""
        if self._s_values is None:
            return []
        return self._s_values.copy()

    def clear(self):
        """Clear all waypoints from the trajectory."""
        self.waypoints = []
        self._s_values = []
        self._interpolators = {}
        self._sampled_s = []
        self._sampled_waypoints = []

    def sample_by_distance(self, interval: float) -> List[Waypoint]:
        """
        Sample points along the trajectory at fixed distance intervals.
        The sampled points will include all original waypoints, and additional points
        will be inserted between original waypoints if their distance is greater than
        the specified interval.

        Args:
            interval: Maximum distance between sampled points

        Returns:
            List of sampled Waypoint objects

        Raises:
            ValueError: If trajectory is empty or interval is invalid
        """
        if not self._interpolators:
            raise ValueError("Trajectory must have at least 2 waypoints for sampling")
        if interval <= 0:
            raise ValueError("Sampling interval must be positive")

        sampled_points = []

        # Add first waypoint
        sampled_points.append(self.waypoints[0])

        # Process each pair of consecutive waypoints
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]

            # Calculate distance between waypoints
            dx = wp2.x - wp1.x
            dy = wp2.y - wp1.y
            segment_length = math.sqrt(dx * dx + dy * dy)

            # If segment is longer than interval, insert points
            if segment_length > interval:
                # Calculate number of points to insert
                n_points = math.ceil(segment_length / interval) - 1

                # Calculate actual interval for even spacing
                actual_interval = segment_length / (n_points + 1)

                # Get start and end s values for this segment
                s1 = self._s_values[i]
                s2 = self._s_values[i + 1]

                # Insert interpolated points
                for j in range(n_points):
                    # Calculate s value for interpolated point
                    s = s1 + (j + 1) * actual_interval * (s2 - s1) / segment_length
                    point = self.interpolate_at_distance(s)
                    sampled_points.append(point)

            # Add the next original waypoint
            sampled_points.append(wp2)

        return sampled_points

    def sample_by_number(self, num_points: int) -> List[Waypoint]:
        """
        Sample a fixed number of evenly spaced points along the trajectory.

        Args:
            num_points: Number of points to sample (must be >= 2)

        Returns:
            List of sampled Waypoint objects

        Raises:
            ValueError: If trajectory is empty or num_points is invalid
        """
        if not self._interpolators:
            raise ValueError("Trajectory must have at least 2 waypoints for sampling")
        if num_points < 2:
            raise ValueError("Number of sample points must be at least 2")

        total_length = self.get_trajectory_length()
        s_values = np.linspace(0, total_length, num_points)

        return [self.interpolate_at_distance(s) for s in s_values]

    def find_lookahead_point(self, pose_x: float, pose_y: float, lookahead_distance: float) -> ProjectedPoint:
        """
        Find a point on the trajectory that is a specified distance ahead of the nearest point to the given pose.

        Args:
            pose_x: X coordinate of the current pose
            pose_y: Y coordinate of the current pose
            lookahead_distance: Desired distance to look ahead along the trajectory

        Returns:
            ProjectedPoint containing the look-ahead point data and path distance

        Raises:
            ValueError: If trajectory is empty, lookahead_distance is invalid, or target point would be beyond trajectory end
        """
        if not self._interpolators:
            raise ValueError("Trajectory must have at least 2 waypoints")
        if lookahead_distance <= 0:
            raise ValueError("Look-ahead distance must be positive")

        # Find nearest point on trajectory
        nearest = self.find_nearest_point(pose_x, pose_y)

        # Calculate target distance along trajectory
        target_s = nearest.s + lookahead_distance

        # If target point would be beyond trajectory end, raise error
        if target_s > self.get_trajectory_length():
            raise ValueError("Look-ahead point would be beyond trajectory end")

        # Get the look-ahead point
        point = self.interpolate_at_distance(target_s)
        return ProjectedPoint(x=point.x, y=point.y, yaw=point.yaw, direction=point.direction, s=target_s)

    def __len__(self) -> int:
        """Return the number of waypoints in the trajectory."""
        return len(self.waypoints)

    def __str__(self) -> str:
        """String representation of the trajectory."""
        return f"Trajectory with {len(self.waypoints)} waypoints, length {self.get_trajectory_length():.2f}m"

    def plot(
        self,
        show_arrows: bool = True,
        show_direction: bool = False,
        show_samples: bool = False,
    ):
        """
        Plot the trajectory using matplotlib.

        Args:
            show_arrows: Whether to show orientation arrows at waypoints
            show_direction: Whether to color-code waypoints by direction
            show_samples: Whether to show discrete sample points
        """
        import matplotlib.pyplot as plt

        if not self.waypoints:
            logger.warning("Trajectory is empty, nothing to plot.")
            return

        # For a smoother plot, sample the trajectory
        if len(self.waypoints) > 2:
            sampled_trajectory = Trajectory(
                config=self.config, waypoints=self.sample_by_distance(self.get_trajectory_length() / 20)
            )
            x_coords = [wp.x for wp in sampled_trajectory.waypoints]
            y_coords = [wp.y for wp in sampled_trajectory.waypoints]
        else:
            x_coords = [wp.x for wp in self.waypoints]
            y_coords = [wp.y for wp in self.waypoints]

        plt.plot(x_coords, y_coords, "-b", label="Trajectory")

        if show_arrows:
            for wp in self.waypoints:
                plt.arrow(
                    wp.x,
                    wp.y,
                    0.5 * math.cos(wp.yaw),
                    0.5 * math.sin(wp.yaw),
                    head_width=0.3,
                    head_length=0.4,
                    fc="k",
                    ec="k",
                )

        if show_direction:
            colors = ["green" if wp.direction == 1 else "red" for wp in self.waypoints]
            plt.scatter(
                [wp.x for wp in self.waypoints],
                [wp.y for wp in self.waypoints],
                c=colors,
                s=50,
                label="Direction (Green: Fwd, Red: Rev)",
            )

        if show_samples and self._sampled_waypoints:
            plt.scatter(
                [wp.x for wp in self._sampled_waypoints],
                [wp.y for wp in self._sampled_waypoints],
                s=10,
                c="orange",
                label="Discrete Samples",
                alpha=0.7,
            )

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title("Trajectory Plot")
        plt.grid(True)
        plt.axis("equal")
        plt.legend()


# Example usage and testing
if __name__ == "__main__":
    # Create a simple circular trajectory for testing
    from modules.planner import TrajectoryConfig
    config = TrajectoryConfig()
    trajectory = Trajectory(config=config)

    # Add waypoints for a quarter circle
    n_points = 20
    radius = 10.0
    for i in range(n_points):
        angle = i * math.pi / 2 / (n_points - 1)  # Quarter circle
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        yaw = angle + math.pi / 2  # Tangent direction
        trajectory.add_waypoint(x, y, yaw, direction=1)

    logger.info(f"Created {trajectory}")

    # Test interpolation
    test_s = trajectory.get_trajectory_length() / 2
    point = trajectory.interpolate_at_distance(test_s)
    logger.info(f"Interpolated point at s={test_s:.2f}: {point}")

    # Test nearest point finding
    query_x, query_y = 5.0, 8.0
    nearest = trajectory.find_nearest_point(query_x, query_y)
    logger.info(f"Nearest point to ({query_x}, {query_y}): {nearest}")

    # Test Frenet coordinates
    frenet = trajectory.get_frenet_coordinates(query_x, query_y)
    logger.info(f"Frenet coordinates: {frenet}")
