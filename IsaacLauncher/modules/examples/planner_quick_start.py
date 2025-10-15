from modules.planner import load_config, VehicleState, Trajectory, VehicleModel, \
    VelocityControllerConfig, VelocityController, PurePursuitController
import math
from matplotlib import pyplot as plt
import numpy as np

def adjust_controller():
    config = load_config()

    # Create vehicle model
    wheelbase = 2.9
    vehicle_model = VehicleModel(config.vehicle)
    # waypoints = [
    #     Waypoint(0, 0, 0, 5),
    #     Waypoint(10, 0, 0, 5),
    #     Waypoint(20, 10, np.pi / 4, 5),
    #     Waypoint(30, 20, np.pi / 2, 5)
    # ]
    trajectory = Trajectory(config.trajectory)
    trajectory.add_waypoint(0, 0, 0, -1)
    trajectory.add_waypoint(-3, -0.2, 0.1, -1)
    # trajectory.add_waypoint(20, 10, np.pi / 4, 5)
    # trajectory.add_waypoint(30, 20, np.pi / 2, 5)

    # Create velocity controller optimized for reverse driving
    velocity_config = VelocityControllerConfig()
    velocity_controller = VelocityController(velocity_config)

    # Create pure pursuit controller
    pure_pursuit_config = config.pure_pursuit

    controller = PurePursuitController(
    wheelbase=wheelbase,
    config=pure_pursuit_config,
    trajectory=trajectory,
    velocity_controller=velocity_controller,
    )

    nearest_point = trajectory.find_nearest_point(0, 0)
    print("nearest_point : ", nearest_point)
    vehicle_model.set_state(
        VehicleState(
            position_x=nearest_point.x,
            position_y=nearest_point.y,
            yaw_angle=nearest_point.yaw,
            velocity=0.0,
            steering_angle=0.0,
        )
    )

    x_history = []
    y_history = []
    for _ in range(5000):
        vehicle_state = vehicle_model.get_state()
        time_step = 0.01
        steering, target_velocity = controller.compute_control(vehicle_state, time_step)
        print("steering : ", steering, " target_velocity", target_velocity)
        vehicle_model.update_with_direct_control([steering, target_velocity],time_step)

        # Store position
        x_history.append(vehicle_state.position_x)
        y_history.append(vehicle_state.position_y)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x_history, y_history, 'r*', label='Vehicle Path')
    trajectory.plot()
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    adjust_controller()