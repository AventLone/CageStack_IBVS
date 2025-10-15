#!/usr/bin/env python3
"""
Test script for vehicle display module
Demonstrates the four-wheel vehicle visualization with steering capabilities
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add PathTracking module to path
# sys.path.append(os.path.join(os.path.dirname(__file__), "PathTracking"))

from modules.base import VehicleDisplay, plot_vehicle_simple


def test_basic_vehicle():
    """Test basic vehicle display"""
    print("Testing basic vehicle display...")

    plt.figure(figsize=(10, 6))

    # Test with different steering angles
    steering_angles = [0, math.pi / 6, math.pi / 4, -math.pi / 6, -math.pi / 4]
    positions = [(i * 6, 0) for i in range(len(steering_angles))]
    colors = ["blue", "green", "red", "orange", "purple"]

    display = VehicleDisplay()

    for i, (pos, steer, color) in enumerate(zip(positions, steering_angles, colors)):
        x, y = pos
        display.plot_vehicle(
            x,
            y,
            0,
            steer,
            body_color=color,
            front_wheel_color="red",
            wheel_color="black",
        )

        # Add steering angle text
        plt.text(
            x,
            y - 3,
            f"{math.degrees(steer):.0f}°",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.title("Vehicle Display Test - Different Steering Angles", fontsize=14)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.show()


def test_vehicle_orientations():
    """Test vehicle at different orientations"""
    print("Testing vehicle orientations...")

    plt.figure(figsize=(10, 8))

    # Test different vehicle orientations
    angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    radius = 5

    display = VehicleDisplay(vehicle_length=3.0, vehicle_width=1.5)

    for i, angle in enumerate(angles):
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        yaw = angle + math.pi / 2  # Point towards center

        # Add some steering for visual interest
        steering = math.sin(angle * 2) * math.pi / 6

        display.plot_vehicle(
            x,
            y,
            yaw,
            steering,
            body_color="blue",
            front_wheel_color="red",
            wheel_color="black",
            alpha=0.8,
        )

    plt.title("Vehicle Display Test - Different Orientations and Steering", fontsize=14)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.show()


def test_simple_function():
    """Test the simple plotting function"""
    print("Testing simple plotting function...")

    plt.figure(figsize=(8, 6))

    # Test simple function
    plot_vehicle_simple(
        0,
        0,
        math.pi / 4,
        math.pi / 6,
        body_color="cyan",
        front_wheel_color="magenta",
        wheel_color="black",
    )

    plt.title("Simple Vehicle Plot Function Test", fontsize=14)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")
    plt.show()


def main():
    """Run all tests"""
    print("Vehicle Display Module Test")
    print("===========================")
    print("Testing the new four-wheel vehicle display with steering visualization")
    print()

    try:
        test_basic_vehicle()
        test_vehicle_orientations()
        test_simple_function()

        print("All tests completed successfully!")
        print("\nFeatures demonstrated:")
        print("✓ Four wheels (front wheels steerable)")
        print("✓ Different steering angles")
        print("✓ Vehicle body and orientation")
        print("✓ Direction arrows")
        print("✓ Configurable colors")
        print("✓ Realistic proportions")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
