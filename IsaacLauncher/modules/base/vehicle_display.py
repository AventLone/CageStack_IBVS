"""
Vehicle display utilities for path tracking visualization
Author: PathTracking
Description: Provides comprehensive vehicle visualization with four wheels,
            steerable front wheels, direction indicators, and support for
            plotting on specific matplotlib axes objects for flexible subplot integration

Note: The vehicle display system uses rear wheel center as the reference point.
      Input coordinates (x, y) represent the rear wheel center position.
"""

import math
from typing import Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from loguru import logger


class VehicleDisplay:
    """
    Vehicle visualization class with four wheels and steering display

    This class provides methods to visualize a vehicle with realistic proportions,
    including four wheels (front wheels can be steered), vehicle body, and
    direction indicators.
    """

    def __init__(
        self,
        vehicle_length: float = 4.5,
        vehicle_width: float = 2.0,
        wheelbase: float = 2.9,
        wheel_length: float = 0.4,
        wheel_width: float = 0.2,
        wheel_track: float = 1.6,
    ):
        """
        Initialize vehicle display parameters

        Args:
            vehicle_length (float): Total vehicle length [m]
            vehicle_width (float): Total vehicle width [m]
            wheelbase (float): Distance between front and rear axles [m]
            wheel_length (float): Wheel length [m]
            wheel_width (float): Wheel width [m]
            wheel_track (float): Distance between left and right wheels [m]
        """
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.wheelbase = wheelbase
        self.wheel_length = wheel_length
        self.wheel_width = wheel_width
        self.wheel_track = wheel_track

        # Calculate distances for proper positioning
        self.rear_overhang = (vehicle_length - wheelbase) / 2
        self.front_overhang = self.rear_overhang

    def plot_vehicle(
        self,
        x: float,
        y: float,
        yaw: float,
        steering_angle: float = 0.0,
        ax: Optional[Axes] = None,
        body_color: str = "none",
        wheel_color: str = "black",
        front_wheel_color: str = "red",
        show_direction_arrow: bool = True,
        arrow_color: str = "green",
        alpha: float = 0.3,
        show_labels: bool = False,
    ) -> None:
        """
        Plot vehicle with four wheels and steering visualization

        Args:
            x (float): Rear wheel center x position [m]
            y (float): Rear wheel center y position [m]
            yaw (float): Vehicle heading angle [rad]
            steering_angle (float): Front wheel steering angle [rad]
            ax (Optional[Axes]): Matplotlib axes to plot on. If None, uses current axes
            body_color (str): Color for vehicle body
            wheel_color (str): Color for rear wheels
            front_wheel_color (str): Color for front wheels (to distinguish steering wheels)
            show_direction_arrow (bool): Whether to show direction arrow
            arrow_color (str): Color for direction arrow
            alpha (float): Transparency level (0-1)
            show_labels (bool): Whether to show labels in legend (default: False)
        """
        # Use provided axes or current axes
        if ax is None:
            ax = plt.gca()

        # Vehicle body outline (rectangle)
        body_corners = self._get_vehicle_body_corners()

        # Wheel positions and shapes
        wheel_positions = self._get_wheel_positions()

        # Create rotation matrix for vehicle orientation
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        # Calculate vehicle center position from rear wheel position
        vehicle_center_x = x + (self.wheelbase / 2) * math.cos(yaw)
        vehicle_center_y = y + (self.wheelbase / 2) * math.sin(yaw)

        # Rotate and translate vehicle body around vehicle center
        rotated_body = self._rotate_and_translate(body_corners, rotation_matrix, vehicle_center_x, vehicle_center_y)

        # Plot vehicle body
        body_label = "Vehicle Body" if show_labels else None
        self._plot_polygon(rotated_body, body_color, alpha, body_label, ax)

        # Plot wheels (relative to rear wheel center)
        self._plot_wheels(
            wheel_positions,
            x,
            y,
            yaw,
            steering_angle,
            wheel_color,
            front_wheel_color,
            alpha,
            ax,
            show_labels,
        )

        # Plot direction arrow at vehicle center
        if show_direction_arrow:
            self._plot_direction_arrow(vehicle_center_x, vehicle_center_y, yaw, arrow_color, alpha, ax)

        # Plot rear wheel center point (reference point)
        ax.plot(x, y, "ko", markersize=3, alpha=alpha)

    def _get_vehicle_body_corners(self) -> np.ndarray:
        """
        Get vehicle body corner coordinates in vehicle frame

        Returns:
            np.ndarray: Array of body corner points [x, y] relative to vehicle center
        """
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2

        corners = np.array(
            [
                [-half_length, -half_width],  # Rear left
                [half_length, -half_width],  # Front left
                [half_length, half_width],  # Front right
                [-half_length, half_width],  # Rear right
                [-half_length, -half_width],  # Close the shape
            ]
        )

        return corners

    def _get_wheel_positions(self) -> dict:
        """
        Get wheel center positions relative to rear wheel center

        Returns:
            dict: Dictionary with wheel positions for each wheel
        """
        # Calculate wheel center positions relative to rear wheel center
        front_x = self.wheelbase  # Front wheels are wheelbase distance ahead
        rear_x = 0.0  # Rear wheels are at the reference point
        left_y = self.wheel_track / 2
        right_y = -self.wheel_track / 2

        return {
            "front_left": np.array([front_x, left_y]),
            "front_right": np.array([front_x, right_y]),
            "rear_left": np.array([rear_x, left_y]),
            "rear_right": np.array([rear_x, right_y]),
        }

    def _get_wheel_shape(self) -> np.ndarray:
        """
        Get wheel shape outline in wheel local frame

        Returns:
            np.ndarray: Wheel outline points [x, y]
        """
        half_length = self.wheel_length / 2
        half_width = self.wheel_width / 2

        wheel_shape = np.array(
            [
                [-half_length, -half_width],
                [half_length, -half_width],
                [half_length, half_width],
                [-half_length, half_width],
                [-half_length, -half_width],
            ]
        )

        return wheel_shape

    def _rotate_and_translate(
        self, points: np.ndarray, rotation_matrix: np.ndarray, tx: float, ty: float
    ) -> np.ndarray:
        """
        Rotate and translate points

        Args:
            points (np.ndarray): Points to transform [N x 2]
            rotation_matrix (np.ndarray): 2x2 rotation matrix
            tx (float): Translation in x
            ty (float): Translation in y

        Returns:
            np.ndarray: Transformed points
        """
        # Apply rotation
        rotated = points @ rotation_matrix.T

        # Apply translation
        rotated[:, 0] += tx
        rotated[:, 1] += ty

        return rotated

    def _plot_polygon(
        self,
        points: np.ndarray,
        color: str,
        alpha: float,
        label: Optional[str] = None,
        ax: Optional[Axes] = None,
    ) -> None:
        """
        Plot a polygon from points

        Args:
            points (np.ndarray): Polygon points [N x 2]
            color (str): Fill color
            alpha (float): Transparency
            label (str, optional): Legend label
            ax (Optional[Axes]): Matplotlib axes to plot on. If None, uses current axes
        """
        if ax is None:
            ax = plt.gca()

        polygon = patches.Polygon(
            points[:-1],
            closed=True,
            facecolor=color,
            edgecolor="black",
            alpha=alpha,
            linewidth=1,
        )
        ax.add_patch(polygon)

        ax.plot([], [], color=color, alpha=alpha, label=label)

    def _plot_wheels(
        self,
        wheel_positions: dict,
        vehicle_x: float,
        vehicle_y: float,
        vehicle_yaw: float,
        steering_angle: float,
        wheel_color: str,
        front_wheel_color: str,
        alpha: float,
        ax: Optional[Axes] = None,
        show_labels: bool = False,
    ) -> None:
        """
        Plot all four wheels with proper steering for front wheels

        Args:
            wheel_positions (dict): Wheel center positions relative to rear wheel center
            vehicle_x (float): Rear wheel center x position
            vehicle_y (float): Rear wheel center y position
            vehicle_yaw (float): Vehicle heading
            steering_angle (float): Front wheel steering angle
            wheel_color (str): Rear wheel color
            front_wheel_color (str): Front wheel color
            alpha (float): Transparency
            ax (plt.Axes, optional): Matplotlib axes to plot on. If None, uses current axes
            show_labels (bool): Whether to show labels in legend
        """
        # Use provided axes or current axes
        if ax is None:
            ax = plt.gca()

        # Vehicle rotation matrix
        cos_yaw = math.cos(vehicle_yaw)
        sin_yaw = math.sin(vehicle_yaw)
        vehicle_rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        # Steering rotation matrix for front wheels
        cos_steer = math.cos(steering_angle)
        sin_steer = math.sin(steering_angle)
        steering_rotation = np.array([[cos_steer, -sin_steer], [sin_steer, cos_steer]])

        wheel_shape = self._get_wheel_shape()

        # Plot each wheel
        for wheel_name, wheel_pos in wheel_positions.items():
            # Determine if this is a front wheel and apply steering
            if "front" in wheel_name:
                # Apply steering rotation first, then vehicle rotation
                rotated_wheel = wheel_shape @ steering_rotation.T @ vehicle_rotation.T
                color = front_wheel_color
                label = "Front Wheels (Steerable)" if wheel_name == "front_left" and show_labels else None
            else:
                # Only apply vehicle rotation for rear wheels
                rotated_wheel = wheel_shape @ vehicle_rotation.T
                color = wheel_color
                label = "Rear Wheels" if wheel_name == "rear_left" and show_labels else None

            # Transform wheel position to world coordinates
            world_wheel_pos = wheel_pos @ vehicle_rotation.T
            wheel_x = vehicle_x + world_wheel_pos[0]
            wheel_y = vehicle_y + world_wheel_pos[1]

            # Translate wheel shape to world position
            rotated_wheel[:, 0] += wheel_x
            rotated_wheel[:, 1] += wheel_y

            # Plot wheel
            self._plot_polygon(rotated_wheel, color, alpha, label, ax)

    def _plot_direction_arrow(
        self,
        x: float,
        y: float,
        yaw: float,
        arrow_color: str,
        alpha: float,
        ax: Optional[Axes] = None,
    ) -> None:
        """
        Plot direction arrow showing vehicle heading

        Args:
            x (float): Vehicle x position
            y (float): Vehicle y position
            yaw (float): Vehicle heading
            arrow_color (str): Arrow color
            alpha (float): Transparency
            ax (plt.Axes, optional): Matplotlib axes to plot on. If None, uses current axes
        """
        # Use provided axes or current axes
        if ax is None:
            ax = plt.gca()

        arrow_length = self.vehicle_length * 0.6
        dx = arrow_length * math.cos(yaw)
        dy = arrow_length * math.sin(yaw)

        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=self.vehicle_width * 0.2,
            head_length=self.vehicle_length * 0.1,
            fc=arrow_color,
            ec=arrow_color,
            alpha=alpha,
            linewidth=2,
        )


def plot_vehicle_simple(
    x: float,
    y: float,
    yaw: float,
    steering_angle: float = 0.0,
    vehicle_length: float = 4.5,
    vehicle_width: float = 2.0,
    ax: Optional[Axes] = None,
    body_color: str = "none",
    alpha: float = 0.3,
    **kwargs,
) -> None:
    """
    Simple function to plot vehicle with default parameters

    Args:
        x (float): Rear wheel center x position [m]
        y (float): Rear wheel center y position [m]
        yaw (float): Vehicle heading [rad]
        steering_angle (float): Steering angle [rad]
        vehicle_length (float): Vehicle length [m]
        vehicle_width (float): Vehicle width [m]
        ax (Optional[Axes]): Matplotlib axes to plot on. If None, uses current axes
        body_color (str): Color for vehicle body
        alpha (float): Transparency level (0-1)
        **kwargs: Additional arguments passed to VehicleDisplay.plot_vehicle()
    """
    display = VehicleDisplay(vehicle_length=vehicle_length, vehicle_width=vehicle_width)
    display.plot_vehicle(x, y, yaw, steering_angle, ax=ax, body_color=body_color, alpha=alpha, **kwargs)


def demo_vehicle_display():
    """
    Demonstration of vehicle display capabilities
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Vehicle Display Demo - Four Wheels with Steering Visualization",
        fontsize=16,
        fontweight="bold",
    )

    # Create vehicle display instance
    display = VehicleDisplay()

    # Demo 1: Straight vehicle
    ax1 = axes[0, 0]
    display.plot_vehicle(
        0,
        0,
        0,
        0,
        ax=ax1,
        body_color="blue",
        front_wheel_color="red",
        wheel_color="black",
    )
    ax1.set_title("Straight Vehicle (0° steering)")
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Demo 2: Left turn
    ax2 = axes[0, 1]
    display.plot_vehicle(
        0,
        0,
        0,
        math.pi / 6,
        ax=ax2,
        body_color="green",
        front_wheel_color="orange",
        wheel_color="darkgray",
    )
    ax2.set_title("Left Turn (30° steering)")
    ax2.axis("equal")
    ax2.grid(True, alpha=0.3)

    # Demo 3: Right turn
    ax3 = axes[1, 0]
    display.plot_vehicle(
        0,
        0,
        0,
        -math.pi / 4,
        ax=ax3,
        body_color="purple",
        front_wheel_color="yellow",
        wheel_color="brown",
    )
    ax3.set_title("Right Turn (-45° steering)")
    ax3.axis("equal")
    ax3.grid(True, alpha=0.3)

    # Demo 4: Vehicle at angle with steering
    ax4 = axes[1, 1]
    display.plot_vehicle(
        0,
        0,
        math.pi / 3,
        math.pi / 8,
        ax=ax4,
        body_color="red",
        front_wheel_color="cyan",
        wheel_color="magenta",
    )
    ax4.set_title("Angled Vehicle (60° heading, 22.5° steering)")
    ax4.axis("equal")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demo_vehicle_trajectory(ax: Optional[Axes] = None):
    """
    Demonstration of vehicle following a trajectory with steering

    Args:
        ax (Optional[Axes]): Matplotlib axes to plot on. If None, creates new figure
    """
    if ax is None:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

    # Create a circular trajectory
    t = np.linspace(0, 2 * math.pi, 20)
    radius = 10
    x_traj = radius * np.cos(t)
    y_traj = radius * np.sin(t)

    # Plot trajectory
    ax.plot(x_traj, y_traj, "k--", linewidth=2, alpha=0.7, label="Trajectory")

    # Create vehicle display
    display = VehicleDisplay(vehicle_length=2.0, vehicle_width=1.5)

    # Plot vehicles at different positions along trajectory
    colors = ["red", "blue", "green", "orange", "purple"]

    for i in range(0, len(t), 4):  # Every 4th point
        x = x_traj[i]
        y = y_traj[i]
        yaw = t[i] + math.pi / 2  # Tangent to circle

        # Calculate steering angle for circular motion
        steering_angle = math.atan(display.wheelbase / radius)

        color = colors[i // 4 % len(colors)]
        display.plot_vehicle(
            x,
            y,
            yaw,
            steering_angle,
            ax=ax,
            body_color=color,
            alpha=0.6,
            show_direction_arrow=True,
        )

    ax.set_title(
        "Vehicle Following Circular Trajectory\n(Shows steering angles for turning)",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Y Position [m]")

    # Only show if we created our own figure
    if ax == plt.gca():
        plt.show()


if __name__ == "__main__":
    logger.info("Vehicle Display Module")
    logger.info("======================")
    logger.info("Features:")
    logger.info("- Four wheel vehicle visualization")
    logger.info("- Steerable front wheels (red/colored)")
    logger.info("- Non-steerable rear wheels (black/gray)")
    logger.info("- Vehicle body with configurable colors")
    logger.info("- Direction arrow showing vehicle heading")
    logger.info("- Realistic vehicle proportions")
    logger.info("\nRunning demonstrations...")

    # Run demonstrations
    demo_vehicle_display()
    demo_vehicle_trajectory()
