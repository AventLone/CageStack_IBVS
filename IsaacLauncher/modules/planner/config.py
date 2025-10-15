"""
Configuration for PathTracking

This module defines dataclasses for configuration of different components.
It provides a simple function to load configuration from a YAML file.

Author: Assistant
"""

import os
import yaml
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Dict, Any, Optional, Type
import numpy as np

# Setup logger with VSCode-clickable format
# setup_logger()

@dataclass
class VehicleConfig:
    """Vehicle model configuration parameters"""

    # Physical parameters
    wheelbase: float = 1.65  # Distance between front and rear axles [m]
    vehicle_length: float = 4.5  # Total vehicle length [m]
    vehicle_width: float = 2.0  # Total vehicle width [m]
    wheel_length: float = 0.4  # Wheel length [m]
    wheel_width: float = 0.2  # Wheel width [m]
    wheel_track: float = 1.6  # Distance between left and right wheels [m]

    # Kinematic limits
    max_steering_angle: float = 45.0  # Maximum steering angle [degrees]
    max_velocity: float = 50.0  # Maximum forward velocity [m/s]
    min_velocity: float = -10.0  # Maximum reverse velocity [m/s] (negative value)
    max_steering_rate: float = 30.0  # Maximum steering rate [degrees/s]
    max_acceleration: float = 3.0  # Maximum acceleration [m/s²]
    max_deceleration: float = 5.0  # Maximum deceleration [m/s²] (positive value)

    # Delay parameters
    steering_delay: float = 0.0  # Time delay for steering commands [s]
    acceleration_delay: float = 0.0  # Time delay for acceleration commands [s]

    # Control gains for direct control
    steering_rate_gain: float = 5.0  # Gain for converting steering error to steering rate
    acceleration_gain: float = 2.0  # Gain for converting velocity error to acceleration

    # Control noise parameters
    control_input_noise_enabled: bool = True  # Enable/disable control input noise (master switch)
    steering_noise_std: float = 0.01  # Standard deviation for steering angle noise [rad]
    noise_seed: Optional[int] = None  # Random seed for reproducible noise (None for random)
    
    # Simple noise parameters (for basic Gaussian noise on true position)
    simple_position_noise_std: float = 0.1  # Standard deviation for simple position noise [m]
    simple_yaw_noise_std: float = 0.02  # Standard deviation for simple yaw angle noise [rad]
    simple_velocity_noise_std: float = 0.05  # Standard deviation for simple velocity noise [m/s]
    
    # Default state type to return when get_state() is called
    default_state_type: str = "true"  # Options: "true", "simple"

    def get_max_steering_angle_rad(self) -> float:
        """Get maximum steering angle in radians"""
        return np.deg2rad(self.max_steering_angle)

    def get_max_steering_rate_rad(self) -> float:
        """Get maximum steering rate in radians/s"""
        return np.deg2rad(self.max_steering_rate)


@dataclass
class PurePursuitConfig:
    """Pure Pursuit controller configuration parameters"""

    min_lookahead: float = 1.5  # Minimum lookahead distance [m]
    k_gain: float = 0.8  # Lookahead distance gain (lookahead = k_gain * velocity + min_lookahead)
    max_steering_angle: float = 35.0  # Maximum steering angle [degrees]
    goal_tolerance: float = 0.01  # Distance tolerance to consider goal reached [m]
    velocity_tolerance: float = 0.1  # Velocity tolerance to consider vehicle stopped [m/s]

    def get_max_steering_angle_rad(self) -> float:
        """Get maximum steering angle in radians"""
        return np.deg2rad(self.max_steering_angle)


@dataclass
class VelocityControllerConfig:
    """Velocity controller configuration parameters"""

    max_forward_velocity: float = 1.5  # Maximum forward velocity [m/s]
    max_backward_velocity: float = 1.0  # Maximum backward velocity [m/s]
    max_acceleration: float = 0.5  # Maximum acceleration magnitude [m/s²]
    max_deceleration: float = 0.5  # Maximum deceleration magnitude [m/s²]
    goal_tolerance: float = 0.01  # Distance tolerance to consider goal reached [m]
    velocity_tolerance: float = 0.1  # Velocity tolerance to consider vehicle stopped [m/s]
    conservative_braking_factor: float = 1.2  # Safety factor for deceleration distance
    min_velocity: float = 0.05  # Minimum velocity magnitude [m/s]
    
    # Simplified ramp down control parameters
    enable_segmented_ramp_down: bool = True  # Enable simplified ramp down control
    final_approach_distance: float = 0.4  # Distance for final approach at minimum velocity [m]


@dataclass
class TrajectoryConfig:
    """Trajectory configuration parameters"""

    discretization_distance: float = 0.1  # Default distance between discretized points [m]
    default_sample_count: int = 100  # Initial default sample count


@dataclass
class SimulationConfig:
    """Simulation configuration parameters"""

    time_step: float = 0.1  # Default simulation time step [s]
    max_time: float = 60.0  # Default maximum simulation time [s]

    # Visualization parameters
    plot_margin: float = 5.0  # Margin around trajectory for plot view [m]
    figure_size: tuple = (16, 8)  # Default figure size (width, height)

    # Animation parameters
    animation_interval: int = 50  # Matplotlib animation interval [ms]


@dataclass
class PathTrackingConfig:
    """Main configuration container for the entire PathTracking project"""

    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    pure_pursuit: PurePursuitConfig = field(default_factory=PurePursuitConfig)
    velocity_controller: VelocityControllerConfig = field(default_factory=VelocityControllerConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def _update_config_from_dict(config_obj: Any, config_dict: Dict[str, Any]) -> None:
    """
    Recursively update dataclass fields from a dictionary.
    """
    for f in fields(config_obj):
        if f.name in config_dict:
            value = config_dict[f.name]
            if is_dataclass(f.type) and isinstance(value, dict):
                _update_config_from_dict(getattr(config_obj, f.name), value)
            else:
                setattr(config_obj, f.name, value)


def load_config(config_file: Optional[str] = None) -> PathTrackingConfig:
    """
    Load PathTracking configuration.

    This function loads configuration from a YAML file and merges it with the default
    configuration. If no file is provided, it returns the default configuration.

    Args:
        config_file (Optional[str]): Path to the YAML configuration file.

    Returns:
        PathTrackingConfig: The fully populated configuration object.
    """
    config = PathTrackingConfig()

    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as f:
            yaml_config = yaml.safe_load(f)

        if yaml_config:
            _update_config_from_dict(config, yaml_config)

    return config
