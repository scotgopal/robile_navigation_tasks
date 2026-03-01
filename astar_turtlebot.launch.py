#!/usr/bin/env python3
"""
Launch file: TurtleBot3 Gazebo + A* path planner node

Usage (from the workspace root):
  python3 astar_turtlebot.launch.py                       # quick test
  ros2 launch astar_turtlebot.launch.py                   # via ros2 launch
  ros2 launch astar_turtlebot.launch.py use_odom:=false   # use AMCL instead
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ── paths ────────────────────────────────────────────────────────────
    this_dir = os.path.dirname(os.path.abspath(__file__))
    map_yaml = os.path.join(this_dir, "maps", "map_task1.yaml")
    map_pgm = os.path.join(this_dir, "maps", "map_task1.pgm")

    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    # Try to use the robile_navigation params if available
    robile_nav_params = os.path.expanduser(
        "~/robile_navigation/config/nav2_params.yaml"
    )
    if not os.path.isfile(robile_nav_params):
        robile_nav_params = os.path.join(
            nav2_bringup_dir, "params", "nav2_params.yaml"
        )

    # ── launch arguments ─────────────────────────────────────────────────
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_odom = LaunchConfiguration("use_odom")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("use_odom", default_value="true",
                                  description="true=use /odom, false=use /amcl_pose"),
            DeclareLaunchArgument("world", default_value="turtlebot3_world",
                                  description="Gazebo world (empty_world / turtlebot3_world / turtlebot3_house)"),

            # ── 1. TurtleBot3 Gazebo simulation ─────────────────────────
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(tb3_gazebo_dir, "launch",
                                 "turtlebot3_world.launch.py")
                ),
                launch_arguments={"use_sim_time": use_sim_time}.items(),
            ),

            # ── 2. Nav2 stack (map_server + AMCL + controllers) ─────────
            #     Provides /map, /amcl_pose, TF tree.
            #     The planner_server/controller_server are launched but we
            #     drive the robot ourselves via /cmd_vel.
            TimerAction(
                period=5.0,  # wait for Gazebo to start
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            os.path.join(nav2_bringup_dir, "launch",
                                         "bringup_launch.py")
                        ),
                        launch_arguments={
                            "use_sim_time": use_sim_time,
                            "map": map_yaml,
                            "params_file": robile_nav_params,
                            "autostart": "true",
                        }.items(),
                    ),
                ],
            ),

            # ── 3. RViz ─────────────────────────────────────────────────
            TimerAction(
                period=8.0,
                actions=[
                    Node(
                        package="rviz2",
                        executable="rviz2",
                        name="rviz2",
                        arguments=[
                            "-d",
                            os.path.join(nav2_bringup_dir, "rviz",
                                         "nav2_default_view.rviz"),
                        ],
                        parameters=[{"use_sim_time": use_sim_time}],
                        output="screen",
                    ),
                ],
            ),

            # ── 4. A* TurtleBot node ────────────────────────────────────
            TimerAction(
                period=10.0,  # give nav2 time to publish /map
                actions=[
                    Node(
                        package=None,         # standalone script
                        executable=None,
                        name="astar_turtlebot",
                        output="screen",
                        parameters=[
                            {
                                "use_sim_time": use_sim_time,
                                "map_pgm": map_pgm,
                                "map_yaml": map_yaml,
                                "use_odom": use_odom,
                                "inflation_radius": 3,
                                "lookahead": 0.35,
                                "linear_speed": 0.18,
                                "goal_tolerance": 0.15,
                                "path_downsample": 3,
                                "control_rate": 10.0,
                            }
                        ],
                    )
                    if False  # placeholder — see ExecuteProcess below
                    else ExecuteProcess(
                        cmd=[
                            "python3",
                            os.path.join(this_dir, "astar_turtlebot_node.py"),
                            "--ros-args",
                            "-p", "use_sim_time:=true",
                            "-p", f"map_pgm:={map_pgm}",
                            "-p", f"map_yaml:={map_yaml}",
                            "-p", ["use_odom:=", use_odom],
                            "-p", "inflation_radius:=3",
                            "-p", "lookahead:=0.35",
                            "-p", "linear_speed:=0.18",
                            "-p", "goal_tolerance:=0.15",
                            "-p", "path_downsample:=3",
                            "-p", "control_rate:=10.0",
                        ],
                        output="screen",
                    ),
                ],
            ),
        ]
    )
