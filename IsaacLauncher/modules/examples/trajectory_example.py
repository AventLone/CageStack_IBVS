"""
Comprehensive examples demonstrating the Trajectory class functionality.

This file shows how to use the Trajectory class for various path tracking scenarios,
including creating different trajectory shapes and using projection/interpolation features.

Author: Assistant
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from modules.planner import (
    Trajectory,
)
from modules.planner.config import load_config

def create_straight_line_trajectory() -> Trajectory:
    """Create a simple straight line trajectory."""
    config = load_config()
    trajectory = Trajectory(config.trajectory)

    # Create a straight line from (0,0) to (10,0)
    for i in range(11):
        x = i
        y = 0
        yaw = 0  # Heading along x-axis
        trajectory.add_waypoint(x, y, yaw, direction=1)

    return trajectory


def create_circular_trajectory(radius: float = 10.0, n_points: int = 50) -> Trajectory:
    """Create a circular trajectory."""
    config = load_config()
    trajectory = Trajectory(config.trajectory)

    for i in range(n_points):
        angle = i * 2 * math.pi / (n_points - 1)  # Full circle
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        yaw = angle + math.pi / 2  # Tangent direction
        trajectory.add_waypoint(x, y, yaw, direction=1)

    return trajectory


def create_s_curve_trajectory() -> Trajectory:
    """Create an S-shaped trajectory."""
    config = load_config()
    trajectory = Trajectory(config.trajectory)

    # Generate S-curve using sine function
    x_coords = []
    y_coords = []
    yaw_angles = []

    for i in range(100):
        t = i / 99.0  # Normalized parameter [0, 1]
        x = t * 20  # Length of 20 units
        y = 5 * math.sin(t * 2 * math.pi)  # S-curve with amplitude 5

        # Calculate heading angle (tangent to curve)
        if i < 99:
            t_next = (i + 1) / 99.0
            x_next = t_next * 20
            y_next = 5 * math.sin(t_next * 2 * math.pi)
            yaw = math.atan2(y_next - y, x_next - x)
        else:
            yaw = yaw_angles[-1]  # Use previous yaw for last point

        x_coords.append(x)
        y_coords.append(y)
        yaw_angles.append(yaw)

    trajectory.add_waypoints_from_arrays(x_coords, y_coords, yaw_angles)
    return trajectory


def create_parking_maneuver_trajectory() -> Trajectory:
    """Create a parking maneuver trajectory with forward and backward directions."""
    config = load_config()
    trajectory = Trajectory(config.trajectory)

    # Forward motion
    for i in range(10):
        x = i
        y = 0
        yaw = 0
        trajectory.add_waypoint(x, y, yaw, direction=1)

    # Turn and reverse into parking space
    for i in range(10):
        t = i / 9.0
        x = 9 + 3 * math.sin(t * math.pi / 2)
        y = 3 * (1 - math.cos(t * math.pi / 2))
        yaw = t * math.pi / 2
        trajectory.add_waypoint(x, y, yaw, direction=-1)  # Reverse direction

    return trajectory


def demonstrate_interpolation(trajectory: Trajectory):
    """Demonstrate interpolation capabilities."""
    print(f"\n=== Interpolation Demo for {trajectory} ===")

    total_length = trajectory.get_trajectory_length()
    print(f"Total trajectory length: {total_length:.2f}")

    # Sample points along the trajectory
    test_distances = [
        0,
        total_length * 0.25,
        total_length * 0.5,
        total_length * 0.75,
        total_length,
    ]

    for s in test_distances:
        try:
            point = trajectory.interpolate_at_distance(s)
            print(
                f"s={s:6.2f}: pos=({point.x:6.2f}, {point.y:6.2f}), " f"yaw={point.yaw:6.2f}, dir={point.direction:2d}"
            )
        except ValueError as e:
            print(f"s={s:6.2f}: Error - {e}")


def demonstrate_projection(trajectory: Trajectory):
    """Demonstrate nearest point projection and Frenet coordinates."""
    print(f"\n=== Projection Demo for {trajectory} ===")

    # Test various query points
    test_points = [
        (0, 0),  # On the trajectory
        (5, 2),  # Slightly off trajectory
        (10, -3),  # Further off trajectory
        (-2, 1),  # Before trajectory start
        (15, 0),  # After trajectory end
    ]

    for pose_x, pose_y in test_points:
        try:
            # Find nearest point
            nearest = trajectory.find_nearest_point(pose_x, pose_y)

            # Get Frenet coordinates
            frenet = trajectory.get_frenet_coordinates(pose_x, pose_y)

            # Calculate actual distance to nearest point
            actual_distance = math.sqrt((pose_x - nearest.x) ** 2 + (pose_y - nearest.y) ** 2)

            print(
                f"Query: ({pose_x:5.1f}, {pose_y:5.1f}) -> "
                f"Nearest: ({nearest.x:5.2f}, {nearest.y:5.2f}), "
                f"s={nearest.s:5.2f}, d={frenet.d:5.2f}, "
                f"dist={actual_distance:.3f}"
            )

        except ValueError as e:
            print(f"Query: ({pose_x:5.1f}, {pose_y:5.1f}) -> Error: {e}")


def plot_trajectory_with_projections(trajectory: Trajectory, test_points: list):
    """Plot trajectory with projection examples (requires matplotlib)."""
    try:
        # Sample trajectory points for plotting
        total_length = trajectory.get_trajectory_length()
        s_samples = np.linspace(0, total_length, 200)

        traj_x = []
        traj_y = []
        for s in s_samples:
            point = trajectory.interpolate_at_distance(s)
            traj_x.append(point.x)
            traj_y.append(point.y)

        plt.figure(figsize=(12, 8))

        # Plot trajectory
        plt.plot(traj_x, traj_y, "b-", linewidth=2, label="Trajectory")

        # Plot waypoints
        waypoint_x = [wp.x for wp in trajectory.waypoints]
        waypoint_y = [wp.y for wp in trajectory.waypoints]
        plt.plot(waypoint_x, waypoint_y, "bo", markersize=4, label="Waypoints")

        # Plot test points and their projections
        for i, (px, py) in enumerate(test_points):
            try:
                nearest = trajectory.find_nearest_point(px, py)

                # Plot query point
                plt.plot(px, py, "ro", markersize=8, label="Query Points" if i == 0 else "")

                # Plot projection line
                plt.plot(
                    [px, nearest.x],
                    [py, nearest.y],
                    "r--",
                    alpha=0.7,
                    label="Projections" if i == 0 else "",
                )

                # Plot projected point
                plt.plot(
                    nearest.x,
                    nearest.y,
                    "gs",
                    markersize=6,
                    label="Projected Points" if i == 0 else "",
                )

                # Add Frenet coordinates as text
                frenet = trajectory.get_frenet_coordinates(px, py)
                plt.annotate(
                    f"s={frenet.s:.1f}\nd={frenet.d:.1f}",
                    xy=(px, py),
                    xytext=(px + 0.5, py + 0.5),
                    fontsize=8,
                    ha="left",
                )

            except ValueError:
                plt.plot(px, py, "rx", markersize=8)

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")
        plt.title("Trajectory with Nearest Point Projections")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.show()

    except ImportError:
        print("Matplotlib not available. Skipping visualization.")


def main():
    """Main demonstration function."""
    print("=== Trajectory Class Demonstration ===")

    # Create different types of trajectories
    trajectories = [
        ("Straight Line", create_straight_line_trajectory()),
        ("Circular", create_circular_trajectory(radius=5.0, n_points=30)),
        ("S-Curve", create_s_curve_trajectory()),
        ("Parking Maneuver", create_parking_maneuver_trajectory()),
    ]

    for name, traj in trajectories:
        print(f"\n{'='*50}")
        print(f"Testing {name} Trajectory")
        print(f"{'='*50}")

        # Demonstrate interpolation
        demonstrate_interpolation(traj)

        # Demonstrate projection
        demonstrate_projection(traj)

    # Detailed example with the S-curve
    print(f"\n{'='*50}")
    print("Detailed S-Curve Analysis")
    print(f"{'='*50}")

    s_curve = create_s_curve_trajectory()

    # Test points for detailed analysis
    test_points = [(5, 3), (10, -2), (15, 4), (0, 0), (30, 0)]

    print("\nDetailed Frenet Coordinate Analysis:")
    for px, py in test_points:
        frenet = s_curve.get_frenet_coordinates(px, py)
        nearest = s_curve.find_nearest_point(px, py)

        print(
            f"Point ({px:3.0f}, {py:3.0f}): "
            f"s={frenet.s:6.2f}, d={frenet.d:6.2f}, "
            f"nearest=({nearest.x:6.2f}, {nearest.y:6.2f}), "
            f"yaw={nearest.yaw:6.2f}"
        )

    # Visualization (if matplotlib is available)
    plot_trajectory_with_projections(s_curve, test_points)

    print(f"\n{'='*50}")
    print("Demonstration Complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
