#!/usr/bin/env python3
"""
Launch TurtleBot3 Gazebo simulation with custom A* global planner.

This mirrors ``nav2_bringup/tb3_simulation_launch.py`` but replaces
the default ``nav2_planner/planner_server`` with our Python A* node
(``astar_planner_server.py``).  The A* node is a plain Node (not a
LifecycleNode), so it is NOT managed by the lifecycle_manager.

Usage
-----
  ros2 launch astar_tb3_launch.py headless:=False
  ros2 launch astar_tb3_launch.py headless:=False map:=/path/to/map.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # ── directories ──────────────────────────────────────────────────
    bringup_dir = get_package_share_directory("nav2_bringup")
    launch_dir  = os.path.join(bringup_dir, "launch")
    this_dir    = os.path.dirname(os.path.abspath(__file__))

    # ── launch configuration variables ───────────────────────────────
    namespace       = LaunchConfiguration("namespace")
    use_sim_time    = LaunchConfiguration("use_sim_time")
    params_file     = LaunchConfiguration("params_file")
    autostart       = LaunchConfiguration("autostart")
    headless        = LaunchConfiguration("headless")
    use_rviz        = LaunchConfiguration("use_rviz")
    slam            = LaunchConfiguration("slam")
    map_yaml_file   = LaunchConfiguration("map")
    world           = LaunchConfiguration("world")
    use_respawn     = LaunchConfiguration("use_respawn")
    log_level       = LaunchConfiguration("log_level")
    robot_name      = LaunchConfiguration("robot_name")
    robot_sdf       = LaunchConfiguration("robot_sdf")
    rviz_config     = LaunchConfiguration("rviz_config_file")

    pose = {
        "x": LaunchConfiguration("x_pose", default="-2.00"),
        "y": LaunchConfiguration("y_pose", default="-0.50"),
        "z": LaunchConfiguration("z_pose", default="0.01"),
        "R": LaunchConfiguration("roll",   default="0.00"),
        "P": LaunchConfiguration("pitch",  default="0.00"),
        "Y": LaunchConfiguration("yaw",    default="0.00"),
    }

    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]

    # ── rewritten params (same trick as nav2_bringup) ────────────────
    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=namespace,
            param_rewrites={
                "use_sim_time": use_sim_time,
                "yaml_filename": map_yaml_file,
                "autostart": autostart,
            },
            convert_types=True,
        ),
        allow_substs=True,
    )

    # Lifecycle nodes managed by lifecycle_manager_navigation.
    # NOTE: "planner_server" is NOT listed here because our A* planner
    # is a plain Node (not a LifecycleNode), so it doesn't need to be
    # managed by the lifecycle_manager.
    lifecycle_nodes = [
        "controller_server",
        "smoother_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
        "velocity_smoother",
    ]

    # ================================================================
    # Declare launch arguments
    # ================================================================
    decls = [
        DeclareLaunchArgument("namespace",      default_value=""),
        DeclareLaunchArgument("use_sim_time",   default_value="true"),
        DeclareLaunchArgument("slam",           default_value="False"),
        DeclareLaunchArgument("params_file",
            default_value=os.path.join(bringup_dir, "params",
                                       "nav2_params.yaml")),
        DeclareLaunchArgument("autostart",      default_value="true"),
        DeclareLaunchArgument("headless",       default_value="True"),
        DeclareLaunchArgument("use_rviz",       default_value="True"),
        DeclareLaunchArgument("use_respawn",    default_value="False"),
        DeclareLaunchArgument("log_level",      default_value="info"),
        DeclareLaunchArgument("map",
            default_value=os.path.join(bringup_dir, "maps",
                                       "turtlebot3_world.yaml")),
        DeclareLaunchArgument("world",
            default_value=os.path.join(bringup_dir, "worlds",
                                       "world_only.model")),
        DeclareLaunchArgument("robot_name",
            default_value="turtlebot3_waffle"),
        DeclareLaunchArgument("robot_sdf",
            default_value=os.path.join(bringup_dir, "worlds",
                                       "waffle.model")),
        DeclareLaunchArgument("rviz_config_file",
            default_value=os.path.join(bringup_dir, "rviz",
                                       "nav2_default_view.rviz")),
    ]

    # ================================================================
    # 1. Gazebo
    # ================================================================
    start_gzserver = ExecuteProcess(
        cmd=["gzserver",
             "-s", "libgazebo_ros_init.so",
             "-s", "libgazebo_ros_factory.so",
             world],
        cwd=[launch_dir],
        output="screen",
    )

    start_gzclient = ExecuteProcess(
        condition=IfCondition(PythonExpression(["not ", headless])),
        cmd=["gzclient"],
        cwd=[launch_dir],
        output="screen",
    )

    # ================================================================
    # 2. Robot state publisher + spawner
    # ================================================================
    urdf_path = os.path.join(bringup_dir, "urdf", "turtlebot3_waffle.urdf")
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    start_robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace=namespace,
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "robot_description": robot_description,
        }],
        remappings=remappings,
    )

    start_spawner = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity",          robot_name,
            "-file",            robot_sdf,
            "-robot_namespace", namespace,
            "-x", pose["x"], "-y", pose["y"], "-z", pose["z"],
            "-R", pose["R"], "-P", pose["P"], "-Y", pose["Y"],
        ],
    )

    # ================================================================
    # 3. Localisation  (map_server + AMCL  — unchanged from nav2)
    # ================================================================
    start_localisation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, "localization_launch.py")),
        condition=IfCondition(PythonExpression(["not ", slam])),
        launch_arguments={
            "namespace":       namespace,
            "map":             map_yaml_file,
            "use_sim_time":    use_sim_time,
            "autostart":       autostart,
            "params_file":     params_file,
            "use_composition": "False",
            "use_respawn":     use_respawn,
            "container_name":  "nav2_container",
        }.items(),
    )

    start_slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, "slam_launch.py")),
        condition=IfCondition(slam),
        launch_arguments={
            "namespace":    namespace,
            "use_sim_time": use_sim_time,
            "autostart":    autostart,
            "use_respawn":  use_respawn,
            "params_file":  params_file,
        }.items(),
    )

    # ================================================================
    # 4. Navigation stack  (same as navigation_launch.py but our planner)
    # ================================================================
    nav_nodes = GroupAction([
        # ---- controller (local planner + local costmap) ----
        Node(
            package="nav2_controller",
            executable="controller_server",
            output="screen",
            respawn=use_respawn,
            respawn_delay=2.0,
            parameters=[configured_params],
            arguments=["--ros-args", "--log-level", log_level],
            remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
        ),
        # ---- smoother ----
        Node(
            package="nav2_smoother",
            executable="smoother_server",
            name="smoother_server",
            output="screen",
            respawn=use_respawn,
            respawn_delay=2.0,
            parameters=[configured_params],
            arguments=["--ros-args", "--log-level", log_level],
            remappings=remappings,
        ),
        # ───────────────────────────────────────────────────────
        #  ★  OUR A* PLANNER  (replaces nav2_planner/planner_server)
        # ───────────────────────────────────────────────────────
        ExecuteProcess(
            cmd=[
                "python3",
                os.path.join(this_dir, "astar_planner_server.py"),
                "--ros-args",
                "-p", "use_sim_time:=true",
                "-p", "inflation_radius_cells:=3",
            ],
            output="screen",
        ),
        # ---- behaviour server (recovery behaviours) ----
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            output="screen",
            respawn=use_respawn,
            respawn_delay=2.0,
            parameters=[configured_params],
            arguments=["--ros-args", "--log-level", log_level],
            remappings=remappings,
        ),
        # ---- BT navigator ----
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            output="screen",
            respawn=use_respawn,
            respawn_delay=2.0,
            parameters=[configured_params],
            arguments=["--ros-args", "--log-level", log_level],
            remappings=remappings,
        ),
        # ---- waypoint follower ----
        Node(
            package="nav2_waypoint_follower",
            executable="waypoint_follower",
            name="waypoint_follower",
            output="screen",
            respawn=use_respawn,
            respawn_delay=2.0,
            parameters=[configured_params],
            arguments=["--ros-args", "--log-level", log_level],
            remappings=remappings,
        ),
        # ---- velocity smoother ----
        Node(
            package="nav2_velocity_smoother",
            executable="velocity_smoother",
            name="velocity_smoother",
            output="screen",
            respawn=use_respawn,
            respawn_delay=2.0,
            parameters=[configured_params],
            arguments=["--ros-args", "--log-level", log_level],
            remappings=remappings + [
                ("cmd_vel", "cmd_vel_nav"),
                ("cmd_vel_smoothed", "cmd_vel"),
            ],
        ),
        # ---- lifecycle manager ----
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            arguments=["--ros-args", "--log-level", log_level],
            parameters=[
                {"use_sim_time": use_sim_time},
                {"autostart": autostart},
                {"node_names": lifecycle_nodes},
            ],
        ),
    ])

    # ================================================================
    # 5. RViz
    # ================================================================
    start_rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, "rviz_launch.py")),
        condition=IfCondition(use_rviz),
        launch_arguments={
            "namespace":     namespace,
            "use_namespace": "false",
            "rviz_config":   rviz_config,
        }.items(),
    )

    # ================================================================
    # Build launch description
    # ================================================================
    ld = LaunchDescription()

    ld.add_action(
        SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"))

    for d in decls:
        ld.add_action(d)

    # Gazebo
    ld.add_action(start_gzserver)
    ld.add_action(start_gzclient)

    # Robot
    ld.add_action(start_robot_state_pub)
    ld.add_action(start_spawner)

    # Localisation
    ld.add_action(start_localisation)
    ld.add_action(start_slam)

    # Navigation (with our A* planner)
    ld.add_action(nav_nodes)

    # RViz
    ld.add_action(start_rviz)

    return ld
