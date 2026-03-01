#!/usr/bin/env python3
"""
Launch file: A* Planner Server (standalone)
============================================
Launches ONLY the A* planner server node.  Nav2 components
(controller_server, bt_navigator, etc.) run separately — either
on the same machine or on another machine on the same ROS 2 network.

Usage
-----
::

    # Terminal 1 — start the A* planner server
    ros2 launch astar_planner_launch.py

    # Terminal 2 — start Nav2 + simulation (planner_server excluded)
    ros2 launch astar_tb3_launch.py

    # Or, run the planner directly without a launch file:
    python3 astar_planner_server.py --ros-args \\
        -p use_sim_time:=true \\
        -p inflation_radius_cells:=3

Cross-machine setup
-------------------
If Nav2 runs on Machine B and this planner runs on Machine A, make
sure both machines share the same ROS_DOMAIN_ID::

    export ROS_DOMAIN_ID=42   # same on both machines

Then bt_navigator on Machine B will automatically discover our
``compute_path_to_pose`` action server on Machine A via DDS.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # ── launch arguments ─────────────────────────────────────────
    use_sim_time = LaunchConfiguration("use_sim_time")
    inflation    = LaunchConfiguration("inflation_radius_cells")
    downsample   = LaunchConfiguration("path_downsample")
    snap_radius  = LaunchConfiguration("snap_radius")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="Use simulation (Gazebo) clock"),
        DeclareLaunchArgument(
            "inflation_radius_cells", default_value="3",
            description="Obstacle inflation radius in grid cells"),
        DeclareLaunchArgument(
            "path_downsample", default_value="1",
            description="Keep every Nth waypoint (1 = no downsampling)"),
        DeclareLaunchArgument(
            "snap_radius", default_value="15",
            description="Max cells to search when snapping to free space"),

        # ── A* Planner Server ────────────────────────────────────
        ExecuteProcess(
            cmd=[
                "python3",
                os.path.join(this_dir, "astar_planner_server.py"),
                "--ros-args",
                "-p", ["use_sim_time:=", use_sim_time],
                "-p", ["inflation_radius_cells:=", inflation],
                "-p", ["path_downsample:=", downsample],
                "-p", ["snap_radius:=", snap_radius],
            ],
            output="screen",
        ),
    ])
