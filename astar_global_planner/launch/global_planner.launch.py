"""Launch the A* Global Planner standalone node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node as LaunchNode


def generate_launch_description():
    pkg_dir = get_package_share_directory('astar_global_planner')

    # ---------- Launch arguments ----------
    map_topic_arg = DeclareLaunchArgument(
        'map_topic', default_value='/map',
        description='OccupancyGrid topic')

    plan_topic_arg = DeclareLaunchArgument(
        'plan_topic', default_value='/plan',
        description='Topic for the published Path')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation clock')

    # ---------- A* planner node ----------
    planner_node = LaunchNode(
        package='astar_global_planner',
        executable='astar_global_planner',
        name='astar_global_planner',
        output='screen',
        parameters=[{
            'map_topic': LaunchConfiguration('map_topic'),
            'plan_topic': LaunchConfiguration('plan_topic'),
            'goal_tolerance': 0.25,
            'use_costmap_weights': True,
            'do_smooth_path': True,
            'visualize_path': True,
            'max_planning_time': 5.0,
            'lethal_threshold': 90,
            'inscribed_threshold': 50,
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )

    return LaunchDescription([
        map_topic_arg,
        plan_topic_arg,
        use_sim_time_arg,
        planner_node,
    ])
