#!/usr/bin/env python3
"""
Occupancy Grid Path Planning with A* Algorithm
Loads map from PNG/PGM + YAML, runs A* between random points, and visualizes result
Includes obstacle avoidance verification and visualization of explored nodes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml
from PIL import Image
import heapq
import random
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
import argparse
import time

@dataclass(order=True)
class PrioritizedItem:
    """Priority queue item for A* algorithm"""
    priority: float
    f_score: float = field(compare=False)
    g_score: float = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    parent: Optional['PrioritizedItem'] = field(compare=False, default=None)

class OccupancyGridMap:
    """Represents and loads an occupancy grid map from image + YAML"""
    
    def __init__(self, yaml_file: str):
        """Load map from YAML file (references image file)"""
        self.yaml_file = yaml_file
        self.map_dir = os.path.dirname(yaml_file)
        
        # Load YAML parameters
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Map parameters
        self.image_file = os.path.join(self.map_dir, config['image'])
        self.resolution = config['resolution']  # meters per pixel
        self.origin = config['origin']  # [x, y, theta]
        
        # Load the image using PIL
        img = Image.open(self.image_file).convert('L')  # Convert to grayscale
        self.original_img = np.array(img)
        
        # Set dimensions FIRST before using them
        self.height, self.width = self.original_img.shape
        
        # Threshold to binary occupancy (0 = free, 1 = occupied)
        # Assuming white (255) is free, black (0) is occupied
        threshold = 128
        self.occupancy = (self.original_img < threshold).astype(np.int8)
        
        # Add border of occupied cells to prevent planning outside map
        self.occupancy[0, :] = 1
        self.occupancy[-1, :] = 1
        self.occupancy[:, 0] = 1
        self.occupancy[:, -1] = 1
        
        # Create a dilated obstacle map for safer planning (NOW height/width are defined)
        self.occupancy_dilated = self._dilate_obstacles(self.occupancy, radius=1)
        
        # Ensure dilated map also has borders
        self.occupancy_dilated[0, :] = 1
        self.occupancy_dilated[-1, :] = 1
        self.occupancy_dilated[:, 0] = 1
        self.occupancy_dilated[:, -1] = 1
        
        print(f"Loaded map: {self.image_file}")
        print(f"Dimensions: {self.width} x {self.height} pixels")
        print(f"Resolution: {self.resolution} m/pixel")
        print(f"Origin: {self.origin}")
        print(f"Occupied cells: {np.sum(self.occupancy)}/{self.width * self.height}")
        print(f"Free cells: {np.sum(self.occupancy == 0)}/{self.width * self.height}")
    
    def _dilate_obstacles(self, occupancy: np.ndarray, radius: int = 1) -> np.ndarray:
        """Simple dilation of obstacles for safer planning"""
        dilated = occupancy.copy()
        if radius <= 0:
            return dilated
        
        # Find obstacle indices
        obstacle_y, obstacle_x = np.where(occupancy == 1)
        
        # Dilate each obstacle
        for y, x in zip(obstacle_y, obstacle_x):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        dilated[ny, nx] = 1
        
        return dilated
    
    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates"""
        px = x # int((x - self.origin[0]) / self.resolution)
        py = y # int((y - self.origin[1]) / self.resolution)
        return px, py
    
    def pixel_to_world(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates"""
        x = px  # * self.resolution + self.origin[0]
        y = py # * self.resolution + self.origin[1]
        return x, y
    
    def is_occupied(self, x: int, y: int, use_dilated: bool = False) -> bool:
        """Check if a pixel is occupied"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True  # Out of bounds treated as occupied
        
        if use_dilated:
            return self.occupancy_dilated[y, x] == 1
        else:
            return self.occupancy[y, x] == 1  # Note: image coordinates (y, x)
    
    def is_free(self, x: int, y: int, use_dilated: bool = False) -> bool:
        """Check if a pixel is free"""
        return not self.is_occupied(x, y, use_dilated)
    
    def get_random_free_cell(self, use_dilated: bool = False) -> Tuple[int, int]:
        """Get a random free cell in the map with y between 50-120"""
        if use_dilated:
            free_cells = np.argwhere(self.occupancy_dilated == 0)
        else:
            free_cells = np.argwhere(self.occupancy == 0)
        
        if len(free_cells) == 0:
            raise ValueError("No free cells in map!")
        
        # Filter for y-coordinate between 50 and 120
        # free_cells format is [[y, x], [y, x], ...] because numpy.argwhere returns (row, col)
        y_min, y_max = 50, 120
        filtered_cells = [cell for cell in free_cells if y_min <= cell[0] <= y_max]
        
        if len(filtered_cells) == 0:
            print(f"Warning: No free cells in y-range [{y_min}, {y_max}]. Using any free cell.")
            idx = random.randint(0, len(free_cells) - 1)
            y, x = free_cells[idx]
        else:
            idx = random.randint(0, len(filtered_cells) - 1)
            y, x = filtered_cells[idx]
        
        return x, y  # Return (x, y) as before
    
    def get_distance_to_nearest_obstacle(self, x: int, y: int, max_dist: int = 10) -> float:
        """Calculate distance to nearest obstacle (for debugging)"""
        if self.is_occupied(x, y):
            return 0.0
        
        min_dist = float('inf')
        for dy in range(-max_dist, max_dist + 1):
            for dx in range(-max_dist, max_dist + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.is_occupied(nx, ny):
                        dist = np.sqrt(dx**2 + dy**2)
                        min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else max_dist

class AStarPlanner:
    """A* path planning algorithm with obstacle avoidance verification"""
    
    def __init__(self, occupancy_map: OccupancyGridMap, use_dilated_map: bool = True):
        self.map = occupancy_map
        self.use_dilated_map = use_dilated_map
        # 8-connected grid movements with costs
        self.movements = [
            (1, 0, 1.0),    # right
            (-1, 0, 1.0),   # left
            (0, 1, 1.0),    # down
            (0, -1, 1.0),   # up
            (1, 1, 1.414),  # diagonal down-right
            (1, -1, 1.414), # diagonal up-right
            (-1, 1, 1.414), # diagonal down-left
            (-1, -1, 1.414) # diagonal up-left
        ]
        
        print(f"Using {'dilated' if use_dilated_map else 'original'} obstacle map")
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic (admissible and consistent)"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        """Get valid neighboring cells and movement costs"""
        neighbors = []
        for dx, dy, cost in self.movements:
            nx, ny = pos[0] + dx, pos[1] + dy
            if self.map.is_free(nx, ny, self.use_dilated_map):
                neighbors.append((nx, ny, cost))
        return neighbors
    
    def is_path_collision_free(self, path: List[Tuple[int, int]]) -> bool:
        """Verify that a path doesn't intersect with obstacles"""
        if not path:
            return False
        
        for point in path:
            if self.map.is_occupied(point[0], point[1]):
                print(f"WARNING: Path point {point} is occupied!")
                return False
        
        # Check line segments between consecutive points
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # Use Bresenham's line algorithm to check all cells on the line
            points = self._get_line_points(x1, y1, x2, y2)
            for px, py in points:
                if self.map.is_occupied(px, py):
                    print(f"WARNING: Path segment from {path[i]} to {path[i+1]} passes through occupied cell ({px}, {py})")
                    return False
        
        return True
    
    def _get_line_points(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm to get all points on a line"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        
        return points
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[Tuple[List[Tuple[int, int]], Dict]]:
        """
        Find path from start to goal using A* algorithm
        Returns tuple of (path, stats) or (None, stats) if no path found
        """
        if self.map.is_occupied(*start, self.use_dilated_map):
            print(f"Start position {start} is occupied!")
            # Find nearest free cell for debugging
            dist = self.map.get_distance_to_nearest_obstacle(*start)
            print(f"Distance to nearest obstacle: {dist:.2f} pixels")
            return None, {}
        
        if self.map.is_occupied(*goal, self.use_dilated_map):
            print(f"Goal position {goal} is occupied!")
            dist = self.map.get_distance_to_nearest_obstacle(*goal)
            print(f"Distance to nearest obstacle: {dist:.2f} pixels")
            return None, {}
        
        # Statistics for debugging
        stats = {
            'nodes_explored': 0,
            'max_queue_size': 0,
            'path_length_pixels': 0,
            'path_length_world': 0,
            'computation_time': 0,
            'explored_nodes': []  # Store explored nodes for visualization
        }
        
        start_time = time.time()
        
        # Priority queue: (f_score, counter, item)
        counter = 0
        open_set = []
        
        # For each cell, the cost of the cheapest path from start
        g_score = {start: 0.0}
        
        # For each cell, the estimated total cost from start to goal through this cell
        f_score = {start: self.heuristic(start, goal)}
        
        # For each cell, the best parent
        came_from = {}
        
        # Track visited cells
        closed_set = set()
        
        # Push start node
        start_item = PrioritizedItem(f_score[start], f_score[start], 0.0, start, None)
        heapq.heappush(open_set, start_item)
        
        while open_set:
            stats['max_queue_size'] = max(stats['max_queue_size'], len(open_set))
            current_item = heapq.heappop(open_set)
            current = current_item.position
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            stats['nodes_explored'] += 1
            
            # Store for visualization (sample every 100th node to avoid memory issues)
            if stats['nodes_explored'] % 100 == 0:
                stats['explored_nodes'].append(current)
            
            # Check if we reached the goal
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                # Calculate statistics
                stats['computation_time'] = time.time() - start_time
                
                # Calculate path length
                for i in range(1, len(path)):
                    dx = path[i][0] - path[i-1][0]
                    dy = path[i][1] - path[i-1][1]
                    stats['path_length_pixels'] += np.sqrt(dx**2 + dy**2)
                    
                    # World coordinates
                    wx1, wy1 = self.map.pixel_to_world(*path[i-1])
                    wx2, wy2 = self.map.pixel_to_world(*path[i])
                    stats['path_length_world'] += np.sqrt((wx2-wx1)**2 + (wy2-wy1)**2)
                
                # Verify path is collision-free
                if self.is_path_collision_free(path):
                    print("✓ Path verification: No collisions detected")
                else:
                    print("✗ WARNING: Path verification found collisions!")
                
                return path, stats
            
            # Explore neighbors
            for nx, ny, move_cost in self.get_neighbors(current):
                neighbor = (nx, ny)
                
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path is better than previous ones
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    
                    # Add to open set
                    counter += 1
                    neighbor_item = PrioritizedItem(f, f, tentative_g, neighbor, current_item)
                    heapq.heappush(open_set, neighbor_item)
        
        # No path found
        stats['computation_time'] = time.time() - start_time
        return None, stats

class MapVisualizer:
    """Visualization tools for occupancy grid and paths"""
    
    def __init__(self, occupancy_map: OccupancyGridMap):
        self.map = occupancy_map
        
    def create_figure(self):
        """Create figure for visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Display occupancy grid (inverted: white = free, black = occupied)
        img_display = 1 - self.map.occupancy  # Invert for display (1=free, 0=occupied)
        
        # Left subplot - Original map
        axes[0].imshow(img_display, cmap='gray', origin='upper',
                      extent=[0, self.map.width, self.map.height, 0])
        axes[0].set_title('Original Map')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Right subplot - Map with path and explored nodes
        axes[1].imshow(img_display, cmap='gray', origin='upper',
                      extent=[0, self.map.width, self.map.height, 0])
        axes[1].set_title('A* Path Planning Result')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        return fig, axes
    
    def plot_explored_nodes(self, ax, nodes: List[Tuple[int, int]], alpha: float = 0.3):
        """Plot explored nodes to show algorithm's search space"""
        if nodes:
            x_coords = [n[0] for n in nodes]
            y_coords = [n[1] for n in nodes]
            ax.scatter(x_coords, y_coords, c='yellow', s=10, alpha=alpha, 
                      label='Explored nodes', zorder=1)
    
    def plot_path(self, ax, path: List[Tuple[int, int]], color: str = 'blue', 
                  linewidth: float = 3, label: str = 'Path'):
        """Plot a path on the axes"""
        if not path:
            return
        
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, color=color, linewidth=linewidth, label=label, zorder=3)
        
        # Also plot path points
        ax.scatter(path_x, path_y, c=color, s=20, alpha=0.7, zorder=2)
    
    def plot_start_goal(self, ax, start: Tuple[int, int], goal: Tuple[int, int]):
        """Plot start and goal markers"""
        ax.plot(start[0], start[1], 'go', markersize=15, label='Start', 
                markeredgecolor='black', markeredgewidth=2, zorder=4)
        ax.plot(goal[0], goal[1], 'ro', markersize=15, label='Goal', 
                markeredgecolor='black', markeredgewidth=2, zorder=4)
    
    def add_legend(self, ax):
        """Add legend to axes"""
        ax.legend(loc='upper right', fontsize=10)
    
    def add_obstacle_highlight(self, ax, path: List[Tuple[int, int]]):
        """Highlight obstacles near the path for verification"""
        if not path:
            return
        
        # Check each path point's distance to obstacles
        for point in path[::5]:  # Sample every 5th point to avoid clutter
            dist = self.map.get_distance_to_nearest_obstacle(point[0], point[1], max_dist=5)
            if dist < 2.0:  # Highlight points very close to obstacles
                ax.plot(point[0], point[1], 'r*', markersize=8, alpha=0.7)

def main():
    parser = argparse.ArgumentParser(description='A* path planning on occupancy grid map')
    parser.add_argument('yaml_file', help='Path to YAML map file')
    parser.add_argument('--start', nargs=2, type=float, default=None,
                        help='Start position in world coordinates (x y)')
    parser.add_argument('--goal', nargs=2, type=float, default=None,
                        help='Goal position in world coordinates (x y)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save', type=str, default=None,
                        help='Save visualization to file')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display the plot (only save if --save is used)')
    parser.add_argument('--use-dilated', action='store_true', default=True,
                        help='Use dilated obstacle map for safer planning')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # Load map
        print("=" * 60)
        print("OCCUPANCY GRID PATH PLANNING WITH A*")
        print("=" * 60)
        occupancy_map = OccupancyGridMap(args.yaml_file)
        
        # Get start and goal positions
        print("\n" + "=" * 60)
        print("SETTING START AND GOAL POSITIONS")
        print("=" * 60)
        
        if args.start is not None:
            start_world = (args.start[0], args.start[1])
            start_pixel = occupancy_map.world_to_pixel(*start_world)
            print(f"Start (world): ({start_world[0]:.2f}, {start_world[1]:.2f})")
        else:
            start_pixel = occupancy_map.get_random_free_cell(args.use_dilated)
            start_world = occupancy_map.pixel_to_world(*start_pixel)
            print(f"Random start (pixel): {start_pixel}")
            print(f"Random start (world): ({start_world[0]:.2f}, {start_world[1]:.2f})")
        
        if args.goal is not None:
            goal_world = (args.goal[0], args.goal[1])
            goal_pixel = occupancy_map.world_to_pixel(*goal_world)
            print(f"Goal (world): ({goal_world[0]:.2f}, {goal_world[1]:.2f})")
        else:
            # Get a different random goal
            goal_pixel = occupancy_map.get_random_free_cell(args.use_dilated)
            while goal_pixel == start_pixel:
                goal_pixel = occupancy_map.get_random_free_cell(args.use_dilated)
            goal_world = occupancy_map.pixel_to_world(*goal_pixel)
            print(f"Random goal (pixel): {goal_pixel}")
            print(f"Random goal (world): ({goal_world[0]:.2f}, {goal_world[1]:.2f})")
        
        # Check if start and goal are free
        print("\n" + "=" * 60)
        print("VALIDATING POSITIONS")
        print("=" * 60)
        
        start_free = occupancy_map.is_free(*start_pixel, args.use_dilated)
        goal_free = occupancy_map.is_free(*goal_pixel, args.use_dilated)
        
        if not start_free:
            print(f"WARNING: Start position {start_pixel} is occupied!")
            # Find nearest free cell
            print("Attempting to find nearest free cell...")
            found = False
            for radius in range(1, 20):
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        nx, ny = start_pixel[0] + dx, start_pixel[1] + dy
                        if occupancy_map.is_free(nx, ny, args.use_dilated):
                            start_pixel = (nx, ny)
                            start_world = occupancy_map.pixel_to_world(nx, ny)
                            print(f"✓ Adjusted start to: {start_pixel} (world: {start_world[0]:.2f}, {start_world[1]:.2f})")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                print("✗ Could not find free start position!")
                return
        else:
            print(f"✓ Start position is free")
        
        if not goal_free:
            print(f"WARNING: Goal position {goal_pixel} is occupied!")
            # Find nearest free cell
            print("Attempting to find nearest free cell...")
            found = False
            for radius in range(1, 20):
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        nx, ny = goal_pixel[0] + dx, goal_pixel[1] + dy
                        if occupancy_map.is_free(nx, ny, args.use_dilated):
                            goal_pixel = (nx, ny)
                            goal_world = occupancy_map.pixel_to_world(nx, ny)
                            print(f"✓ Adjusted goal to: {goal_pixel} (world: {goal_world[0]:.2f}, {goal_world[1]:.2f})")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                print("✗ Could not find free goal position!")
                return
        else:
            print(f"✓ Goal position is free")
        
        # Check distance to obstacles
        start_obs_dist = occupancy_map.get_distance_to_nearest_obstacle(*start_pixel)
        goal_obs_dist = occupancy_map.get_distance_to_nearest_obstacle(*goal_pixel)
        print(f"Start distance to obstacle: {start_obs_dist:.2f} pixels")
        print(f"Goal distance to obstacle: {goal_obs_dist:.2f} pixels")
        
        # Run A* planning
        print("\n" + "=" * 60)
        print("RUNNING A* PATH PLANNING")
        print("=" * 60)
        planner = AStarPlanner(occupancy_map, use_dilated_map=args.use_dilated)
        path, stats = planner.plan(start_pixel, goal_pixel)
        
        if path is None:
            print("\n✗ NO PATH FOUND!")
            print(f"  Nodes explored: {stats.get('nodes_explored', 0)}")
            print(f"  Computation time: {stats.get('computation_time', 0):.3f} seconds")
        else:
            print(f"\n✓ PATH FOUND!")
            print(f"  Nodes explored: {stats['nodes_explored']}")
            print(f"  Max queue size: {stats['max_queue_size']}")
            print(f"  Computation time: {stats['computation_time']:.3f} seconds")
            print(f"  Path length: {len(path)} waypoints")
            print(f"  Path length (pixels): {stats['path_length_pixels']:.2f} pixels")
            print(f"  Path length (world): {stats['path_length_world']:.2f} meters")
            
            # Verify each path point is free
            all_free = all(occupancy_map.is_free(x, y) for x, y in path)
            print(f"  All path points free: {'✓' if all_free else '✗'}")
            
            if args.debug and len(path) > 0:
                print("\nFirst 10 path points:")
                for i, point in enumerate(path[:10]):
                    dist = occupancy_map.get_distance_to_nearest_obstacle(*point)
                    print(f"  {i}: {point} (dist to obstacle: {dist:.2f})")
        
        # Visualize
        print("\n" + "=" * 60)
        print("VISUALIZING RESULTS")
        print("=" * 60)
        visualizer = MapVisualizer(occupancy_map)
        fig, axes = visualizer.create_figure()
        
        # Plot on both subplots
        for ax in axes:
            visualizer.plot_start_goal(ax, start_pixel, goal_pixel)
            
            if path is not None:
                # Plot explored nodes on the right subplot only
                if stats.get('explored_nodes'):
                    visualizer.plot_explored_nodes(axes[1], stats['explored_nodes'], alpha=0.2)
                
                # Plot path
                visualizer.plot_path(ax, path, color='blue', linewidth=3, label='A* Path')
                
                # Highlight obstacle proximity (on right subplot only)
                if args.debug:
                    visualizer.add_obstacle_highlight(axes[1], path)
            
            visualizer.add_legend(ax)
        
        # Set titles with information
        if path:
            path_status = f"✓ PATH FOUND ({len(path)} steps, {stats['path_length_world']:.2f}m)"
        else:
            path_status = "✗ NO PATH FOUND"
        
        axes[0].set_title(f'Original Map\nStart: {start_pixel} | Goal: {goal_pixel}')
        axes[1].set_title(f'A* Path Planning Result\n{path_status}\nNodes explored: {stats.get("nodes_explored", 0)}')
        
        plt.suptitle(f'A* Algorithm - Obstacle Avoidance Verification', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save or show
        if args.save:
            plt.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {args.save}")
        
        if not args.no_display:
            plt.show()
        else:
            plt.close()
        
        print("\n" + "=" * 60)
        print("PLANNING COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()