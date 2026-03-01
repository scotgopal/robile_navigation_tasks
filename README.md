# robile_navigation_tasks

This repository contains simple navigation utilities and demonstrations for mobile robots.

## A* Demo

The `maps/simple.py` script implements a minimal A* planner.

- Running `python3 maps/simple.py` will load `maps/map_task1.pgm` (if present) and perform A* planning on the occupancy grid.
- When the PGM is available, the script automatically selects the first free cell from the top-left as the start and the first free cell from the bottom-right as the goal.
- Without a map file it falls back to a small randomly generated example.

The grid display is rendered in ASCII in the terminal, and step-by-step directions are printed at the end.

Feel free to modify the start/goal selection or supply your own map files.  The loader uses Pillow and NumPy to parse PGM images.

## A* with TurtleBot3 (ROS 2)

`astar_turtlebot_node.py` is a standalone ROS 2 node that uses the same simple
A* algorithm to drive a TurtleBot3 in Gazebo.

### How it works

1. Loads the occupancy map from `maps/map_task1.pgm` (or subscribes to `/map`)
2. Subscribes to `/odom` (or `/amcl_pose`) for the robot's current pose
3. Listens for goals on `/goal_pose` (RViz → **2D Goal Pose** button)
4. Runs A* to compute a collision-free path (with obstacle inflation)
5. Publishes the path on `/planned_path` for visualisation in RViz
6. Follows the path using a pure-pursuit controller, sending `/cmd_vel`

### Quick start (all-in-one launch)

```bash
# Terminal 1 — launch Gazebo + Nav2 + RViz + A* node
export TURTLEBOT3_MODEL=waffle
ros2 launch astar_turtlebot.launch.py
```

Then click **2D Goal Pose** in RViz and place a goal on the map.

### Manual start (step by step)

```bash
# Terminal 1 — TurtleBot3 Gazebo
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2 — Nav2 (provides /map, AMCL, TF)
ros2 launch nav2_bringup bringup_launch.py \
    use_sim_time:=true \
    map:=$(pwd)/maps/map_task1.yaml \
    autostart:=true

# Terminal 3 — A* planner node
python3 astar_turtlebot_node.py --ros-args \
    -p use_sim_time:=true \
    -p map_pgm:=$(pwd)/maps/map_task1.pgm \
    -p use_odom:=true

# Terminal 4 — RViz
rviz2
# Add displays: Map (/map), Path (/planned_path), TF, RobotModel
# Use "2D Goal Pose" to send goals
```

### Parameters

| Parameter          | Default | Description                                    |
|--------------------|---------|------------------------------------------------|
| `map_pgm`          | auto    | Path to the `.pgm` map file                    |
| `map_yaml`         | auto    | Path to the `.yaml` map metadata               |
| `use_odom`         | `true`  | `true` = `/odom`, `false` = `/amcl_pose`       |
| `inflation_radius` | `3`     | Obstacle inflation in grid cells               |
| `lookahead`        | `0.35`  | Pure-pursuit lookahead distance (m)            |
| `linear_speed`     | `0.18`  | Forward speed (m/s)                            |
| `angular_speed_max`| `1.0`   | Max angular velocity (rad/s)                   |
| `goal_tolerance`   | `0.15`  | Distance to goal to declare "reached" (m)      |
| `path_downsample`  | `5`     | Publish every N-th waypoint to `/planned_path` |
| `control_rate`     | `10.0`  | Control loop frequency (Hz)                    |

### Dependencies

- ROS 2 Humble
- `turtlebot3_gazebo`, `nav2_bringup`
- Python: `Pillow`, `numpy`, `PyYAML`, `tf_transformations`

---

## A* integrated into the Nav2 pipeline

`astar_planner_server.py` is a **drop-in replacement** for the default
`nav2_planner/planner_server`.  It is a Python `LifecycleNode` that serves the
`ComputePathToPose` action using our simple A*, so the full Nav2 behaviour tree
(planning → path following → recovery) works exactly as usual — just with A* as
the global planner.

### Architecture

```
RViz "Nav2 Goal"
       │
       ▼
 bt_navigator  ──(ComputePathToPose)──►  astar_planner_server (A*)
       │                                        │
       │                                   reads /map
       │
       └──────(FollowPath)──────────────►  controller_server (DWB)
                                                │
                                           local costmap + lidar
                                                │
                                           cmd_vel → TurtleBot
```

### Quick start

```bash
export TURTLEBOT3_MODEL=waffle

# Launch everything (Gazebo + AMCL + Nav2 + A* planner + RViz):
cd /home/suhail/workspace/amrproject/robile_navigation_tasks
ros2 launch astar_tb3_launch.py headless:=False
```

Then in RViz:
1. Click **2D Pose Estimate** and place the robot's initial pose on the map
2. Click **Nav2 Goal** and place a goal — the A* planner computes a path and
   the DWB controller drives the TurtleBot to it

### What each file does

| File | Role |
|------|------|
| `astar_planner_server.py` | Lifecycle node serving `ComputePathToPose` via A* |
| `astar_tb3_launch.py` | Launch: Gazebo + localisation + nav2 stack + our A* planner + RViz |
| `maps/simple.py` | Original standalone A* demo (no ROS) |

### How it differs from the default Nav2 planner

| | Default (`NavfnPlanner`) | Ours (`astar_planner_server`) |
|---|---|---|
| Language | C++ plugin | Python lifecycle node |
| Algorithm | Dijkstra / optional A* | Simple 8-connected A* |
| Costmap | Uses the full global costmap (inflation + obstacles + sensor layers) | Uses the static `/map` with own inflation |
| Plugin system | Loaded by `planner_server` via `pluginlib` | Replaces `planner_server` entirely |

### Parameters

Pass via `--ros-args -p name:=value` when running standalone:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inflation_radius_cells` | `3` | Obstacle inflation radius (grid cells) |
| `use_sim_time` | `true` | Use Gazebo sim clock |
