"""
Performance Diagnostics Module

This module defines data structures and a class for collecting, analyzing,
and visualizing performance diagnostics for path tracking controllers.
It includes:
- DiagnosticData: A dataclass for storing detailed performance metrics
  at each simulation time step.
- PerformanceDiagnostics: A class that manages the collection,
  analysis, and plotting of historical diagnostic data.
"""

import csv
import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Add the parent directory to the path so we can import modules from PathTracking
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .pure_pursuit import PurePursuitController
from .vehicle_model import VehicleState


@dataclass
class DiagnosticData:
    """
    Data structure for storing diagnostic information at each time step.
    """

    # Time information
    time: float = 0.0

    # Vehicle state
    position_x: float = 0.0
    position_y: float = 0.0
    yaw_angle: float = 0.0
    actual_velocity: float = 0.0
    actual_steering_angle: float = 0.0

    # Control commands
    commanded_velocity: float = 0.0
    commanded_steering_angle: float = 0.0

    # Control computation data
    lookahead_distance: float = 0.0
    target_point_x: float = 0.0
    target_point_y: float = 0.0
    path_direction: float = 0.0
    robot_direction: float = 0.0
    direction_conflict: bool = False

    # Performance metrics
    distance_to_goal: float = 0.0
    stopping_distance: float = 0.0
    max_velocity_for_distance: float = 0.0
    current_acceleration: float = 0.0

    # Error metrics
    longitudinal_error: float = 0.0
    lateral_error: float = 0.0
    angular_error: float = 0.0

    # Status flags
    goal_reached: bool = False


class PerformanceDiagnostics:
    """
    Performance diagnostics system for Pure Pursuit controller.

    Collects and analyzes historical data including:
    - Commanded vs actual velocities and steering angles
    - Control performance metrics
    - Error tracking and analysis
    - Real-time visualization capabilities
    """

    def __init__(self, max_history_size: int = 1000):
        """
        Initialize the performance diagnostics system.

        Args:
            max_history_size (int): Maximum number of data points to store
        """
        self.max_history_size = max_history_size
        self.history: deque = deque(maxlen=max_history_size)
        self.start_time: Optional[float] = None

        # Performance statistics
        self.stats = {
            "velocity_tracking_error": {"mean": 0.0, "std": 0.0, "max": 0.0},
            "steering_tracking_error": {"mean": 0.0, "std": 0.0, "max": 0.0},
            "lateral_error": {"mean": 0.0, "std": 0.0, "max": 0.0},
            "angular_error": {"mean": 0.0, "std": 0.0, "max": 0.0},
            "direction_conflicts": 0,
            "total_distance": 0.0,
            "average_velocity": 0.0,
            "max_acceleration": 0.0,
            "max_deceleration": 0.0,
        }

    def add_data_point(
        self,
        time: float,
        vehicle_state: VehicleState,
        commanded_velocity: float,
        commanded_steering: float,
        controller: PurePursuitController,
        dt: float = 0.1,
    ) -> None:
        """
        Add a new data point to the diagnostic history.

        Args:
            time (float): Current simulation time
            vehicle_state (VehicleState): Current vehicle state
            commanded_velocity (float): Commanded velocity
            commanded_steering (float): Commanded steering angle
            controller (PurePursuitController): Pure pursuit controller instance
            dt (float): Time step
        """
        if self.start_time is None:
            self.start_time = time

        # Get trajectory and compute additional diagnostic data
        trajectory = controller.get_trajectory()
        if trajectory is None:
            return

        # Get target point information
        target_x, target_y, path_direction = (0.0, 0.0, 1.0)
        robot_direction = 1.0
        direction_conflict = False

        target_point = controller.find_target_point(vehicle_state)
        if target_point is not None:
            target_x, target_y, path_direction = target_point
            robot_direction = controller.determine_driving_direction(vehicle_state, target_x, target_y, path_direction)
            direction_conflict = abs(path_direction - robot_direction) > 0.1

        # Calculate performance metrics
        goal_waypoint = trajectory.waypoints[-1]
        dx_goal = vehicle_state.position_x - goal_waypoint.x
        dy_goal = vehicle_state.position_y - goal_waypoint.y
        distance_to_goal = math.sqrt(dx_goal * dx_goal + dy_goal * dy_goal)

        stopping_distance = controller.velocity_controller.calculate_stopping_distance(vehicle_state.velocity)
        current_is_forward = path_direction > 0
        max_vel_for_distance = controller.velocity_controller.calculate_max_velocity_for_distance(
            distance_to_goal, current_is_forward
        )

        current_acceleration = controller.velocity_controller.calculate_current_acceleration(
            vehicle_state.velocity, commanded_velocity, dt
        )

        lookahead_distance = controller.calculate_lookahead_distance(vehicle_state.velocity)

        # Calculate errors
        longitudinal_error, lateral_error, angular_error = controller.calculate_goal_errors(
            vehicle_state, goal_waypoint
        )

        goal_reached = controller.is_goal_reached(vehicle_state)

        # Create diagnostic data point
        data_point = DiagnosticData(
            time=time,
            position_x=vehicle_state.position_x,
            position_y=vehicle_state.position_y,
            yaw_angle=vehicle_state.yaw_angle,
            actual_velocity=vehicle_state.velocity,
            actual_steering_angle=vehicle_state.steering_angle,
            commanded_velocity=commanded_velocity,
            commanded_steering_angle=commanded_steering,
            lookahead_distance=lookahead_distance,
            target_point_x=target_x,
            target_point_y=target_y,
            path_direction=path_direction,
            robot_direction=robot_direction,
            direction_conflict=direction_conflict,
            distance_to_goal=distance_to_goal,
            stopping_distance=stopping_distance,
            max_velocity_for_distance=max_vel_for_distance,
            current_acceleration=current_acceleration,
            longitudinal_error=longitudinal_error,
            lateral_error=lateral_error,
            angular_error=angular_error,
            goal_reached=goal_reached,
        )

        self.history.append(data_point)

        # Update statistics
        self._update_statistics()

    def _update_statistics(self) -> None:
        """Update performance statistics based on current history."""
        if len(self.history) < 2:
            return

        # Convert history to arrays for analysis
        data_points = list(self.history)

        # Velocity tracking error
        velocity_errors = [abs(dp.commanded_velocity - dp.actual_velocity) for dp in data_points]
        if velocity_errors:
            self.stats["velocity_tracking_error"]["mean"] = np.mean(velocity_errors)
            self.stats["velocity_tracking_error"]["std"] = np.std(velocity_errors)
            self.stats["velocity_tracking_error"]["max"] = np.max(velocity_errors)

        # Steering tracking error
        steering_errors = [abs(dp.commanded_steering_angle - dp.actual_steering_angle) for dp in data_points]
        if steering_errors:
            self.stats["steering_tracking_error"]["mean"] = np.mean(steering_errors)
            self.stats["steering_tracking_error"]["std"] = np.std(steering_errors)
            self.stats["steering_tracking_error"]["max"] = np.max(steering_errors)

        # Lateral error
        lateral_errors = [abs(dp.lateral_error) for dp in data_points]
        if lateral_errors:
            self.stats["lateral_error"]["mean"] = np.mean(lateral_errors)
            self.stats["lateral_error"]["std"] = np.std(lateral_errors)
            self.stats["lateral_error"]["max"] = np.max(lateral_errors)

        # Angular error
        angular_errors = [abs(dp.angular_error) for dp in data_points]
        if angular_errors:
            self.stats["angular_error"]["mean"] = np.mean(angular_errors)
            self.stats["angular_error"]["std"] = np.std(angular_errors)
            self.stats["angular_error"]["max"] = np.max(angular_errors)

        # Direction conflicts
        self.stats["direction_conflicts"] = sum(1 for dp in data_points if dp.direction_conflict)

        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(data_points)):
            dx = data_points[i].position_x - data_points[i - 1].position_x
            dy = data_points[i].position_y - data_points[i - 1].position_y
            total_distance += math.sqrt(dx * dx + dy * dy)
        self.stats["total_distance"] = total_distance

        # Average velocity
        velocities = [abs(dp.actual_velocity) for dp in data_points]
        if velocities:
            self.stats["average_velocity"] = np.mean(velocities)

        # Max acceleration/deceleration
        accelerations = [dp.current_acceleration for dp in data_points]
        if accelerations:
            self.stats["max_acceleration"] = np.max(accelerations)
            self.stats["max_deceleration"] = np.min(accelerations)

    def get_diagnostic_summary(self) -> str:
        """
        Generate a comprehensive diagnostic summary.

        Returns:
            str: Formatted diagnostic summary
        """
        if not self.history:
            return "No diagnostic data available"

        summary = "🔍 PERFORMANCE DIAGNOSTICS SUMMARY\n"
        summary += "=" * 50 + "\n\n"

        # Basic statistics
        summary += f"📊 Basic Statistics:\n"
        summary += f"  Data Points: {len(self.history)}\n"
        summary += f"  Total Distance: {self.stats['total_distance']:.2f} m\n"
        summary += f"  Average Velocity: {self.stats['average_velocity']:.2f} m/s\n"
        summary += f"  Max Acceleration: {self.stats['max_acceleration']:.2f} m/s²\n"
        summary += f"  Max Deceleration: {self.stats['max_deceleration']:.2f} m/s²\n\n"

        # Control tracking performance
        summary += f"🎯 Control Tracking Performance:\n"
        summary += f"  Velocity Tracking Error:\n"
        summary += f"    Mean: {self.stats['velocity_tracking_error']['mean']:.3f} m/s\n"
        summary += f"    Std:  {self.stats['velocity_tracking_error']['std']:.3f} m/s\n"
        summary += f"    Max:  {self.stats['velocity_tracking_error']['max']:.3f} m/s\n"
        summary += f"  Steering Tracking Error:\n"
        summary += f"    Mean: {math.degrees(self.stats['steering_tracking_error']['mean']):.2f}°\n"
        summary += f"    Std:  {math.degrees(self.stats['steering_tracking_error']['std']):.2f}°\n"
        summary += f"    Max:  {math.degrees(self.stats['steering_tracking_error']['max']):.2f}°\n\n"

        # Path tracking performance
        summary += f"🛣️  Path Tracking Performance:\n"
        summary += f"  Lateral Error:\n"
        summary += f"    Mean: {self.stats['lateral_error']['mean']:.3f} m\n"
        summary += f"    Std:  {self.stats['lateral_error']['std']:.3f} m\n"
        summary += f"    Max:  {self.stats['lateral_error']['max']:.3f} m\n"
        summary += f"  Angular Error:\n"
        summary += f"    Mean: {math.degrees(self.stats['angular_error']['mean']):.2f}°\n"
        summary += f"    Std:  {math.degrees(self.stats['angular_error']['std']):.2f}°\n"
        summary += f"    Max:  {math.degrees(self.stats['angular_error']['max']):.2f}°\n\n"

        # Direction conflicts
        summary += f"🔄 Direction Analysis:\n"
        summary += f"  Direction Conflicts: {self.stats['direction_conflicts']}\n"
        if len(self.history) > 0:
            conflict_rate = (self.stats["direction_conflicts"] / len(self.history)) * 100
            summary += f"  Conflict Rate: {conflict_rate:.1f}%\n\n"

        return summary

    def plot_diagnostic_charts(self, save_path: Optional[str] = None) -> None:
        """
        Generate comprehensive diagnostic charts.

        Args:
            save_path (str, optional): Path to save the charts. If None, displays interactively.
        """
        if not self.history:
            logger.warning("No diagnostic data available for plotting")
            return

        # Convert history to arrays for plotting
        data_points = list(self.history)
        times = [dp.time for dp in data_points]

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("Pure Pursuit Controller Performance Diagnostics", fontsize=16, fontweight="bold")

        # 1. Velocity tracking
        ax1 = axes[0, 0]
        actual_velocities = [dp.actual_velocity for dp in data_points]
        commanded_velocities = [dp.commanded_velocity for dp in data_points]
        ax1.plot(times, actual_velocities, "b-", label="Actual Velocity", linewidth=2)
        ax1.plot(times, commanded_velocities, "r--", label="Commanded Velocity", linewidth=2)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Velocity (m/s)")
        ax1.set_title("Velocity Tracking Performance")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Steering tracking
        ax2 = axes[0, 1]
        actual_steering = [math.degrees(dp.actual_steering_angle) for dp in data_points]
        commanded_steering = [math.degrees(dp.commanded_steering_angle) for dp in data_points]
        ax2.plot(times, actual_steering, "b-", label="Actual Steering", linewidth=2)
        ax2.plot(times, commanded_steering, "r--", label="Commanded Steering", linewidth=2)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Steering Angle (°)")
        ax2.set_title("Steering Tracking Performance")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Path tracking errors
        ax3 = axes[1, 0]
        lateral_errors = [dp.lateral_error for dp in data_points]
        longitudinal_errors = [dp.longitudinal_error for dp in data_points]
        ax3.plot(times, lateral_errors, "g-", label="Lateral Error", linewidth=2)
        ax3.plot(times, longitudinal_errors, "orange", label="Longitudinal Error", linewidth=2)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Position Error (m)")
        ax3.set_title("Path Tracking Errors")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Control performance metrics
        ax4 = axes[1, 1]
        accelerations = [dp.current_acceleration for dp in data_points]
        lookahead_distances = [dp.lookahead_distance for dp in data_points]
        ax4_twin = ax4.twinx()

        line1 = ax4.plot(times, accelerations, "purple", label="Acceleration", linewidth=2)
        line2 = ax4_twin.plot(times, lookahead_distances, "cyan", label="Lookahead Distance", linewidth=2)

        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Acceleration (m/s²)", color="purple")
        ax4_twin.set_ylabel("Lookahead Distance (m)", color="cyan")
        ax4.set_title("Control Performance Metrics")

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc="upper left")
        ax4.grid(True, alpha=0.3)

        # 5. Direction conflicts and goal distance
        ax5 = axes[2, 0]
        distances_to_goal = [dp.distance_to_goal for dp in data_points]
        stopping_distances = [dp.stopping_distance for dp in data_points]
        ax5.plot(times, distances_to_goal, "red", label="Distance to Goal", linewidth=2)
        ax5.plot(times, stopping_distances, "orange", label="Stopping Distance", linewidth=2)
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Distance (m)")
        ax5.set_title("Goal Approach Analysis")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Tracking error statistics
        ax6 = axes[2, 1]
        velocity_errors = [abs(dp.commanded_velocity - dp.actual_velocity) for dp in data_points]
        steering_errors = [abs(dp.commanded_steering_angle - dp.actual_steering_angle) for dp in data_points]
        steering_errors_deg = [math.degrees(err) for err in steering_errors]

        ax6_twin = ax6.twinx()
        line1 = ax6.plot(times, velocity_errors, "blue", label="Velocity Error", linewidth=2)
        line2 = ax6_twin.plot(times, steering_errors_deg, "red", label="Steering Error", linewidth=2)

        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Velocity Error (m/s)", color="blue")
        ax6_twin.set_ylabel("Steering Error (°)", color="red")
        ax6.set_title("Tracking Error Analysis")

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc="upper left")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Diagnostic charts saved to: {save_path}")
        else:
            plt.show()

    def export_data_to_csv(self, filename: str) -> None:
        """
        Export diagnostic data to CSV file.

        Args:
            filename (str): Output CSV filename (can be just filename or full path)
        """
        if not self.history:
            logger.warning("No diagnostic data available for export")
            return

        # If filename doesn't contain path separator, save to log/csv directory
        if not os.path.dirname(filename):
            # Create log/csv directory if it doesn't exist
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log", "csv")
            os.makedirs(log_dir, exist_ok=True)
            filename = os.path.join(log_dir, filename)

        # Define CSV headers
        headers = [
            "time",
            "position_x",
            "position_y",
            "yaw_angle",
            "actual_velocity",
            "actual_steering_angle",
            "commanded_velocity",
            "commanded_steering_angle",
            "lookahead_distance",
            "target_point_x",
            "target_point_y",
            "path_direction",
            "robot_direction",
            "direction_conflict",
            "distance_to_goal",
            "stopping_distance",
            "max_velocity_for_distance",
            "current_acceleration",
            "longitudinal_error",
            "lateral_error",
            "angular_error",
            "goal_reached",
        ]

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for dp in self.history:
                row = [
                    dp.time,
                    dp.position_x,
                    dp.position_y,
                    dp.yaw_angle,
                    dp.actual_velocity,
                    dp.actual_steering_angle,
                    dp.commanded_velocity,
                    dp.commanded_steering_angle,
                    dp.lookahead_distance,
                    dp.target_point_x,
                    dp.target_point_y,
                    dp.path_direction,
                    dp.robot_direction,
                    dp.direction_conflict,
                    dp.distance_to_goal,
                    dp.stopping_distance,
                    dp.max_velocity_for_distance,
                    dp.current_acceleration,
                    dp.longitudinal_error,
                    dp.lateral_error,
                    dp.angular_error,
                    dp.goal_reached,
                ]
                writer.writerow(row)

        logger.success(f"Diagnostic data exported to: {filename}")

    def load_from_csv(self, filename: str) -> bool:
        """
        Load diagnostic data from CSV file.

        Args:
            filename (str): Input CSV filename

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(filename):
            logger.error(f"Error: File '{filename}' not found")
            return False

        try:
            # Clear existing history
            self.history.clear()
            self.start_time = None

            with open(filename, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    # Convert string values to appropriate types
                    data_point = DiagnosticData(
                        time=float(row["time"]),
                        position_x=float(row["position_x"]),
                        position_y=float(row["position_y"]),
                        yaw_angle=float(row["yaw_angle"]),
                        actual_velocity=float(row["actual_velocity"]),
                        actual_steering_angle=float(row["actual_steering_angle"]),
                        commanded_velocity=float(row["commanded_velocity"]),
                        commanded_steering_angle=float(row["commanded_steering_angle"]),
                        lookahead_distance=float(row["lookahead_distance"]),
                        target_point_x=float(row["target_point_x"]),
                        target_point_y=float(row["target_point_y"]),
                        path_direction=float(row["path_direction"]),
                        robot_direction=float(row["robot_direction"]),
                        direction_conflict=row["direction_conflict"].lower() == "true",
                        distance_to_goal=float(row["distance_to_goal"]),
                        stopping_distance=float(row["stopping_distance"]),
                        max_velocity_for_distance=float(row["max_velocity_for_distance"]),
                        current_acceleration=float(row["current_acceleration"]),
                        longitudinal_error=float(row["longitudinal_error"]),
                        lateral_error=float(row["lateral_error"]),
                        angular_error=float(row["angular_error"]),
                        goal_reached=row["goal_reached"].lower() == "true",
                    )

                    self.history.append(data_point)

                    # Set start time from first data point
                    if self.start_time is None:
                        self.start_time = data_point.time

            # Update statistics after loading
            self._update_statistics()

            logger.success(f"Successfully loaded {len(self.history)} data points from: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return False

    def display_loaded_data(
        self, show_summary: bool = True, show_charts: bool = True, save_chart_path: Optional[str] = None
    ) -> None:
        """
        Display loaded diagnostic data with summary and charts.

        Args:
            show_summary (bool): Whether to print the diagnostic summary
            show_charts (bool): Whether to display diagnostic charts
            save_chart_path (str, optional): Path to save charts instead of displaying
        """
        if not self.history:
            logger.warning("No diagnostic data available to display")
            return

        logger.info(f"\n📁 Loaded Diagnostic Data Display")
        logger.info("=" * 50)
        logger.info(f"Data Points: {len(self.history)}")
        logger.info(f"Time Range: {self.history[0].time:.2f}s - {self.history[-1].time:.2f}s")
        logger.info(f"Duration: {self.history[-1].time - self.history[0].time:.2f}s")

        if show_summary:
            logger.info("\n" + self.get_diagnostic_summary())

        if show_charts:
            logger.info("\n📊 Generating diagnostic charts...")
            self.plot_diagnostic_charts(save_path=save_chart_path)

    @classmethod
    def load_and_display(
        cls,
        filename: str,
        show_summary: bool = True,
        show_charts: bool = True,
        save_chart_path: Optional[str] = None,
        max_history_size: int = 1000,
    ) -> Optional["PerformanceDiagnostics"]:
        """
        Class method to load CSV data and display it in one step.

        Args:
            filename (str): Input CSV filename
            show_summary (bool): Whether to print the diagnostic summary
            show_charts (bool): Whether to display diagnostic charts
            save_chart_path (str, optional): Path to save charts instead of displaying
            max_history_size (int): Maximum number of data points to store

        Returns:
            PerformanceDiagnostics: Instance with loaded data, or None if loading failed
        """
        diagnostics = cls(max_history_size=max_history_size)

        if diagnostics.load_from_csv(filename):
            diagnostics.display_loaded_data(show_summary, show_charts, save_chart_path)
            return diagnostics
        else:
            return None
