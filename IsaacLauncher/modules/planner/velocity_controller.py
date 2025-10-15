"""
Velocity planning controller for path tracking with physics-based velocity planning.

This module contains the VelocityController class that manages target velocity based on:
- Maximum velocity constraints (forward/backward)
- Maximum acceleration/deceleration physics
- Distance to goal for precise stopping at target
- Trajectory direction (forward/backward)
- Minimum velocity constraint
"""

import math
from typing import TYPE_CHECKING, Optional
from loguru import logger

if TYPE_CHECKING:
    from .trajectory import Trajectory
    from .vehicle_model import VehicleState
    from .config import VelocityControllerConfig


class VelocityController:
    """
    Velocity controller for path tracking with physics-based velocity planning.

    This controller manages target velocity based on:
    - Maximum velocity constraints (forward/backward)
    - Maximum acceleration/deceleration physics
    - Distance to goal for precise stopping at target
    - Trajectory direction (forward/backward)
    - Minimum velocity constraint
    """

    def __init__(
        self,
        config: Optional["VelocityControllerConfig"] = None,
    ) -> None:
        """
        Initialize the velocity controller with physics-based acceleration/deceleration.

        Args:
            config (VelocityControllerConfig, optional): Configuration object. If None, creates a default config.
        """
        if config is None:
            # This allows creating a default controller when no config is passed.
            # In a real application, it's recommended to always pass a config.
            from .config import VelocityControllerConfig
            config = VelocityControllerConfig()

        self.config = config

        # Validate velocity constraints
        assert self.config.max_forward_velocity > 0, "max_forward_velocity must be positive"
        # assert self.config.max_backward_velocity < 0, "max_backward_velocity must be negative"
        assert abs(self.config.min_velocity) > 0, "min_velocity must be non-zero"
        assert abs(self.config.min_velocity) <= abs(self.config.max_forward_velocity), "min_velocity cannot exceed max_forward_velocity"
        assert abs(self.config.min_velocity) <= abs(self.config.max_backward_velocity), "min_velocity cannot exceed max_backward_velocity"

        # Validate acceleration parameters
        assert self.config.max_acceleration > 0, "max_acceleration must be positive"
        assert self.config.max_deceleration > 0, "max_deceleration must be positive"
        assert self.config.conservative_braking_factor >= 1.0, "conservative_braking_factor must be >= 1.0"

        # Validate tolerance parameters
        assert self.config.goal_tolerance > 0, "goal_tolerance must be positive"
        assert self.config.velocity_tolerance > 0, "velocity_tolerance must be positive"

        # Validate segmented control parameters if enabled
        if self.config.enable_segmented_ramp_down:
            assert self.config.final_approach_distance > 0, "final_approach_distance must be positive"

        # Store validated parameters
        self.max_forward_velocity = self.config.max_forward_velocity
        self.max_backward_velocity = self.config.max_backward_velocity
        self.max_acceleration = abs(self.config.max_acceleration)
        self.max_deceleration = abs(self.config.max_deceleration)
        self.goal_tolerance = self.config.goal_tolerance
        self.velocity_tolerance = self.config.velocity_tolerance
        self.conservative_braking_factor = self.config.conservative_braking_factor
        self.min_velocity = abs(self.config.min_velocity)
        
        # Simplified ramp down control parameters
        self.enable_segmented_ramp_down = self.config.enable_segmented_ramp_down
        self.final_approach_distance = self.config.final_approach_distance

        logger.info("=== Velocity Controller Configuration ===")
        logger.info(f"Max Forward Velocity: {self.max_forward_velocity} m/s")
        logger.info(f"Max Backward Velocity: {self.max_backward_velocity} m/s")
        logger.info(f"Min Velocity: {self.min_velocity} m/s")
        logger.info(f"Max Acceleration: {self.max_acceleration} m/s²")
        logger.info(f"Max Deceleration: {self.max_deceleration} m/s²")
        logger.info(f"Goal Tolerance: {self.goal_tolerance} m")
        logger.info(f"Velocity Tolerance: {self.velocity_tolerance} m/s")
        logger.info(f"Conservative Braking Factor: {self.conservative_braking_factor}")
        
        if self.enable_segmented_ramp_down:
            logger.info("Simplified Ramp Down Control: Enabled")
            logger.info(f"Final Approach Distance: {self.final_approach_distance} m")
        else:
            logger.info("Simplified Ramp Down Control: Disabled")
        logger.info("=" * 40)

    def calculate_stopping_distance(self, current_velocity: float) -> float:
        """
        Calculate the minimum distance required to stop from current velocity.

        Uses physics equation: d = v²/(2*a) where v is velocity, a is deceleration

        Args:
            current_velocity (float): Current velocity magnitude [m/s]

        Returns:
            float: Stopping distance [m]
        """
        velocity_magnitude = abs(current_velocity)
        if velocity_magnitude < self.velocity_tolerance:
            return 0.0

        # Physics-based stopping distance: d = v²/(2*a)
        stopping_distance = (velocity_magnitude**2) / (2 * self.max_deceleration)

        # Apply safety margin
        return stopping_distance * self.conservative_braking_factor

    def calculate_max_velocity_for_distance(self, distance_to_goal: float, is_forward: bool) -> float:
        """
        Calculate maximum velocity that allows stopping within the given distance.

        Uses physics equation: v = sqrt(2*a*d) where a is deceleration, d is distance

        Args:
            distance_to_goal (float): Distance to goal [m]
            is_forward (bool): Whether motion is forward direction

        Returns:
            float: Maximum velocity magnitude [m/s]
        """
        if distance_to_goal <= self.goal_tolerance:
            return 0.0

        # Account for goal tolerance in available stopping distance
        available_distance = max(0.0, distance_to_goal - self.goal_tolerance)

        # Remove safety margin to get actual usable distance
        usable_distance = available_distance / self.conservative_braking_factor

        if usable_distance <= 0.0:
            return 0.0

        # Physics-based maximum velocity: v = sqrt(2*a*d)
        max_velocity_for_distance = math.sqrt(2 * self.max_deceleration * usable_distance)

        # Apply velocity constraints based on direction
        max_velocity = self.max_forward_velocity if is_forward else self.max_backward_velocity
        return max(min(max_velocity_for_distance, max_velocity), self.min_velocity)

    def is_goal_reached(self, vehicle_state: "VehicleState", trajectory: "Trajectory") -> bool:
        """
        Check if the vehicle has reached the goal (end of trajectory).
        The goal is considered reached if either:
        1. The vehicle is within goal_tolerance distance of the goal point, or
        2. The vehicle's longitudinal position s equals the goal's s on the trajectory

        Args:
            vehicle_state (VehicleState): Current vehicle state
            trajectory (Trajectory): Path trajectory

        Returns:
            bool: True if goal is reached, False otherwise
        """
        if len(trajectory.waypoints) == 0:
            logger.debug("Goal reached: Empty trajectory")
            return True

        # Get the last waypoint (goal position)
        goal_waypoint = trajectory.waypoints[-1]

        # Calculate distance to goal
        dx = vehicle_state.position_x - goal_waypoint.x
        dy = vehicle_state.position_y - goal_waypoint.y
        distance_to_goal = math.sqrt(dx * dx + dy * dy)

        # Check if within distance tolerance
        position_reached = distance_to_goal <= self.goal_tolerance

        # Check if current longitudinal position s equals goal's s
        nearest_point = trajectory.find_nearest_point(vehicle_state.position_x, vehicle_state.position_y)
        trajectory_length = trajectory.get_trajectory_length()
        s_reached = abs(nearest_point.s - trajectory_length) <= self.goal_tolerance

        return position_reached or s_reached

    def calculate_distance_to_goal(self, vehicle_state: "VehicleState", trajectory: "Trajectory") -> float:
        """
        Calculate the distance from the vehicle to the goal (end of trajectory).

        Args:
            vehicle_state (VehicleState): Current vehicle state.
            trajectory (Trajectory): Path trajectory.

        Returns:
            float: Distance to the goal [m].
        """
        if len(trajectory.waypoints) == 0:
            return 0.0

        # Calculate distance to end of trajectory
        trajectory_length = trajectory.get_trajectory_length()
        nearest_point = trajectory.find_nearest_point(vehicle_state.position_x, vehicle_state.position_y)
        distance_to_end = trajectory_length - nearest_point.s

        # Calculate direct distance to goal
        goal_waypoint = trajectory.waypoints[-1]
        dx = vehicle_state.position_x - goal_waypoint.x
        dy = vehicle_state.position_y - goal_waypoint.y
        direct_distance_to_goal = math.sqrt(dx * dx + dy * dy)

        # Use the smaller of the two distances for conservative approach
        return min(distance_to_end, direct_distance_to_goal)

    def compute_segmented_target_velocity(
        self, 
        vehicle_state: "VehicleState", 
        trajectory: "Trajectory", 
        target_direction: float, 
        dt: float = 0.1
    ) -> float:
        """
        Compute target velocity using simplified ramp down control strategy.
        
        This method implements a simplified two-phase control strategy:
        1. Deceleration phase: Uniform deceleration to minimum velocity before final approach distance
        2. Final approach phase: Constant minimum velocity for the final approach distance
        
        Args:
            vehicle_state (VehicleState): Current vehicle state
            trajectory (Trajectory): Path trajectory
            target_direction (float): Direction of motion (1.0 for forward, -1.0 for backward)
            dt (float): Time step for acceleration calculation [s]
            
        Returns:
            float: Target velocity [m/s] (positive for forward, negative for backward)
        """
        # Check if goal is reached
        if self.is_goal_reached(vehicle_state, trajectory):
            logger.debug("Goal reached - returning zero velocity")
            return 0.0
        
        # Calculate distance to goal
        distance_to_goal = self.calculate_distance_to_goal(vehicle_state, trajectory)
        logger.debug(f"Distance to goal: {distance_to_goal:.3f}m")
        
        # Determine motion direction
        is_forward = target_direction > 0
        direction_str = "forward" if is_forward else "backward"
        logger.debug(f"Motion direction: {direction_str}")
        
        # Use configured final approach distance
        final_approach_distance = self.final_approach_distance
        
        if distance_to_goal <= final_approach_distance:
            # Final approach phase: Constant minimum velocity
            desired_velocity = self.min_velocity * target_direction
            logger.debug(f"Final approach phase - using min velocity: {desired_velocity:.3f}m/s")
            
        else:
            # Deceleration phase: Calculate velocity to reach minimum velocity at final approach distance
            # Distance available for deceleration
            deceleration_distance = distance_to_goal - final_approach_distance
            logger.debug(f"Deceleration phase - distance: {deceleration_distance:.3f}m")
            
            # Calculate required velocity to decelerate to minimum velocity over available distance
            # Using physics: v² = v₀² + 2*a*d, where v = min_velocity, a = -max_deceleration
            # Solving for v₀: v₀ = sqrt(v² + 2*a*d)
            min_velocity_squared = self.min_velocity ** 2
            deceleration_term = 2 * self.max_deceleration * deceleration_distance
            required_velocity_magnitude = math.sqrt(min_velocity_squared + deceleration_term)
            
            # Apply velocity constraints based on direction
            max_velocity = self.max_forward_velocity if is_forward else self.max_backward_velocity
            target_velocity_magnitude = min(required_velocity_magnitude, max_velocity)
            
            # Ensure we don't go below minimum velocity
            target_velocity_magnitude = max(target_velocity_magnitude, self.min_velocity)
            
            desired_velocity = target_velocity_magnitude * target_direction
            logger.debug(f"Deceleration phase - target velocity: {desired_velocity:.3f}m/s")
            
        return desired_velocity
    def calculate_current_acceleration(self, current_velocity: float, target_velocity: float, dt: float = 0.1) -> float:
        """
        Calculate the current acceleration based on velocity change.

        Args:
            current_velocity (float): Current velocity [m/s]
            target_velocity (float): Target velocity [m/s]
            dt (float): Time step [s]

        Returns:
            float: Current acceleration [m/s²]
        """
        if dt <= 0:
            return 0.0

        acceleration = (target_velocity - current_velocity) / dt
        return acceleration

    def compute_target_velocity(
        self, vehicle_state: "VehicleState", trajectory: "Trajectory", target_direction: float, dt: float = 0.1
    ) -> float:
        """
        Compute target velocity using physics-based acceleration/deceleration planning.
        
        This method will use segmented ramp down control if enabled, otherwise
        it will use the traditional single-phase control strategy.

        Args:
            vehicle_state (VehicleState): Current vehicle state.
            trajectory (Trajectory): Path trajectory.
            target_direction (float): Direction of motion (1.0 for forward, -1.0 for backward).
            dt (float): Time step for acceleration calculation [s].

        Returns:
            float: Target velocity [m/s] (positive for forward, negative for backward).
        """
        
        return self.compute_segmented_target_velocity(vehicle_state, trajectory, target_direction, dt)

    def get_control_diagnostics(
        self, vehicle_state: "VehicleState", trajectory: "Trajectory", target_direction: float
    ) -> dict:
        """
        Get diagnostic information about the current control state.
        
        This method provides detailed information about the simplified control
        strategy for debugging and monitoring purposes.
        
        Args:
            vehicle_state (VehicleState): Current vehicle state
            trajectory (Trajectory): Path trajectory
            target_direction (float): Direction of motion
            
        Returns:
            dict: Diagnostic information containing:
                - control_phase: Current control phase ('deceleration' or 'final_approach')
                - distance_to_goal: Distance to goal [m]
                - is_forward: Direction of motion
                - segmented_control_enabled: Whether segmented control is enabled
                - final_approach_distance: Distance for final approach phase [m]
        """
        distance_to_goal = self.calculate_distance_to_goal(vehicle_state, trajectory)
        is_forward = target_direction > 0
        
        diagnostics = {
            'segmented_control_enabled': self.enable_segmented_ramp_down,
            'distance_to_goal': distance_to_goal,
            'is_forward': is_forward,
            'current_velocity': vehicle_state.velocity,
            'final_approach_distance': self.final_approach_distance,
        }
        
        if self.enable_segmented_ramp_down:
            # Simplified phase determination
            if distance_to_goal <= self.final_approach_distance:
                control_phase = 'final_approach'
            else:
                control_phase = 'deceleration'
            
            diagnostics.update({
                'control_phase': control_phase,
            })
        else:
            diagnostics.update({
                'control_phase': 'traditional',
            })
        
        return diagnostics

    def print_control_status(
        self, vehicle_state: "VehicleState", trajectory: "Trajectory", target_direction: float
    ) -> None:
        """
        Print current control status for debugging purposes.
        
        Args:
            vehicle_state (VehicleState): Current vehicle state
            trajectory (Trajectory): Path trajectory
            target_direction (float): Direction of motion
        """
        diagnostics = self.get_control_diagnostics(vehicle_state, trajectory, target_direction)
        
        logger.info("=== Velocity Control Status ===")
        logger.info(f"Segmented Control: {'Enabled' if diagnostics['segmented_control_enabled'] else 'Disabled'}")
        logger.info(f"Control Phase: {diagnostics['control_phase']}")
        logger.info(f"Distance to Goal: {diagnostics['distance_to_goal']:.3f} m")
        logger.info(f"Current Velocity: {diagnostics['current_velocity']:.3f} m/s")
        logger.info(f"Direction: {'Forward' if diagnostics['is_forward'] else 'Backward'}")
        
        if diagnostics['segmented_control_enabled']:
            logger.info(f"Final Approach Distance: {diagnostics['final_approach_distance']:.3f} m")
        
        logger.info("=" * 30)
