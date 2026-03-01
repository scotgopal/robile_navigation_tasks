#!/usr/bin/env python3
"""
Simplified A* Path Planning Algorithm
A clear, minimal implementation for learning purposes
"""

import os
import heapq
import math
import random
from typing import List, Tuple, Dict, Set, Optional

# ============================================================================
# PART 1: THE GRID (Simple 2D grid with obstacles)
# ============================================================================

class SimpleGrid:
    """A simple grid with obstacles (0=free, 1=obstacle)"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Create empty grid (all free)
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
    
    def add_obstacle(self, x: int, y: int):
        """Add an obstacle at (x, y)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = 1
    
    def add_random_obstacles(self, num_obstacles: int):
        """Add random obstacles to the grid"""
        for _ in range(num_obstacles):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            self.grid[y][x] = 1
    
    def is_free(self, x: int, y: int) -> bool:
        """Check if a cell is free (not an obstacle and within bounds)"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.grid[y][x] == 0
    
    def display(self, path: List[Tuple[int, int]] = None, start: Tuple[int, int] = None, goal: Tuple[int, int] = None):
        """Display the grid with path, start, and goal"""
        # Create a display grid
        display_grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        
        # Mark obstacles
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 1:
                    display_grid[y][x] = '█'
        
        # Mark path
        if path:
            for x, y in path:
                if display_grid[y][x] == '.':
                    display_grid[y][x] = '○'
        
        # Mark start and goal
        if start:
            x, y = start
            display_grid[y][x] = 'S'
        if goal:
            x, y = goal
            display_grid[y][x] = 'G'
        
        # Print the grid
        print('-' * (self.width + 2))
        for row in display_grid:
            print('|' + ''.join(row) + '|')
        print('-' * (self.width + 2))
        print('Legend: █=Obstacle  .=Free  ○=Path  S=Start  G=Goal')


# ---------------------------------------------------------------------------
# UTILITIES FOR LOADING MAP FILES
# ---------------------------------------------------------------------------

def load_grid_from_pgm(file_path: str, yaml_path: str = None) -> SimpleGrid:
    """Load a PGM occupancy map and convert to SimpleGrid.

    Uses the standard ROS map_server convention:
      occupancy = (255 - pixel_value) / 255
      - pixel ~254 (white)  → free
      - pixel ~0   (black)  → occupied / obstacle
      - pixel ~205 (grey)   → unknown

    If a YAML file is provided (or found next to the PGM) the
    ``occupied_thresh`` and ``free_thresh`` values are read from it.
    Otherwise sensible defaults are used.

    Cells that are *not* free (i.e. occupied **or** unknown) are
    marked as obstacles so the planner routes only through known
    free space.
    """
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is required to load PGM maps. install via pip.")

    import numpy as np

    img = Image.open(file_path)
    arr = np.array(img)
    if arr.ndim != 2:
        raise ValueError("Only single-channel PGM maps are supported.")

    height, width = arr.shape

    # --- read thresholds from the companion YAML if available ---
    occupied_thresh = 0.65
    free_thresh = 0.196          # default: only truly white cells are free
    negate = False

    if yaml_path is None:
        # try to find a .yaml next to the .pgm
        base, _ = os.path.splitext(file_path)
        candidate = base + '.yaml'
        if os.path.isfile(candidate):
            yaml_path = candidate

    if yaml_path is not None:
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                cfg = yaml.safe_load(f)
            occupied_thresh = cfg.get('occupied_thresh', occupied_thresh)
            free_thresh = cfg.get('free_thresh', free_thresh)
            negate = bool(cfg.get('negate', 0))
        except Exception:
            pass   # fall back to defaults

    # --- convert pixel values to occupancy [0..1] ---
    # ROS convention (negate=0): occ = (255 - pixel) / 255
    if negate:
        occ = arr.astype(float) / 255.0
    else:
        occ = (255.0 - arr.astype(float)) / 255.0

    # --- classify cells ---
    #   free:     occ <= free_thresh      (high pixel value, e.g. 254)
    #   occupied: occ >= occupied_thresh  (low pixel value,  e.g. 0)
    #   unknown:  everything in between   (e.g. pixel 205)
    # For safe planning we only allow *free* cells.
    free_mask = occ <= free_thresh    # True where the cell is free

    grid = SimpleGrid(width, height)
    for y in range(height):
        for x in range(width):
            if not free_mask[y, x]:
                grid.add_obstacle(x, y)

    n_free = int(free_mask.sum())
    n_obs = width * height - n_free
    print(f"  Map size : {width} x {height}")
    print(f"  Free cells     : {n_free}")
    print(f"  Obstacle/unknown: {n_obs}")
    return grid


def pick_endpoints(grid: SimpleGrid) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Choose a start and goal position automatically (left → right).

    1. Find the largest connected free-space component (BFS) so both
       points are guaranteed to be reachable from each other.
    2. Pick the leftmost free cell as *start* and the rightmost as *goal*.
    """
    from collections import deque

    # --- BFS flood-fill to find the biggest connected component ---
    visited: Set[Tuple[int, int]] = set()
    best_comp: List[Tuple[int, int]] = []

    for sy in range(grid.height):
        for sx in range(grid.width):
            if grid.is_free(sx, sy) and (sx, sy) not in visited:
                q = deque([(sx, sy)])
                visited.add((sx, sy))
                comp: List[Tuple[int, int]] = []
                while q:
                    cx, cy = q.popleft()
                    comp.append((cx, cy))
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nx, ny = cx + dx, cy + dy
                            if (nx, ny) not in visited and grid.is_free(nx, ny):
                                visited.add((nx, ny))
                                q.append((nx, ny))
                if len(comp) > len(best_comp):
                    best_comp = comp

    if not best_comp:
        raise ValueError("No free cells found in the map")

    # --- leftmost / rightmost inside the main component ---
    best_comp.sort(key=lambda p: (p[0], p[1]))
    start = best_comp[0]    # leftmost (then topmost)
    goal  = best_comp[-1]   # rightmost (then bottommost)

    return start, goal


# ---------------------------------------------------------------------------
# SIMPLE GUI DISPLAY (using Pillow)
# ---------------------------------------------------------------------------

def display_gui(grid: SimpleGrid, path: List[Tuple[int, int]] = None,
                start: Tuple[int, int] = None, goal: Tuple[int, int] = None,
                save_path: str = None):
    """Show the map with obstacles, path, start, and goal in a GUI window.

    * Obstacles  = dark grey
    * Unknown    = light grey background
    * Free space = white
    * Path       = bright red thick line
    * Start      = green circle
    * Goal       = blue circle

    The image is also saved to *save_path* (default ``maps/astar_result.png``)
    so it can be opened in any viewer.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise RuntimeError("Pillow is required for GUI display")

    # --- scale factor so the window is comfortably large ---
    scale = max(2, 800 // max(grid.width, grid.height))
    w, h = grid.width * scale, grid.height * scale

    img = Image.new('RGB', (w, h), (220, 220, 220))   # light-grey background
    draw = ImageDraw.Draw(img)

    # draw free cells white, obstacles dark
    for gy in range(grid.height):
        for gx in range(grid.width):
            x0, y0 = gx * scale, gy * scale
            x1, y1 = x0 + scale - 1, y0 + scale - 1
            if grid.grid[gy][gx] == 1:
                draw.rectangle([x0, y0, x1, y1], fill=(40, 40, 40))
            else:
                draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

    # draw path as a thick red polyline
    if path and len(path) >= 2:
        scaled_path = [(x * scale + scale // 2, y * scale + scale // 2)
                       for x, y in path]
        line_width = max(2, scale // 2)
        draw.line(scaled_path, fill=(255, 40, 40), width=line_width)

    # draw start marker (green filled circle)
    marker_r = max(3, scale)
    if start:
        cx, cy = start[0] * scale + scale // 2, start[1] * scale + scale // 2
        draw.ellipse([cx - marker_r, cy - marker_r, cx + marker_r, cy + marker_r],
                     fill=(0, 200, 0), outline=(0, 100, 0))
    # draw goal marker (blue filled circle)
    if goal:
        cx, cy = goal[0] * scale + scale // 2, goal[1] * scale + scale // 2
        draw.ellipse([cx - marker_r, cy - marker_r, cx + marker_r, cy + marker_r],
                     fill=(0, 80, 255), outline=(0, 40, 150))

    # save to disk so it can always be inspected
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__) or '.', 'astar_result.png')
    img.save(save_path)
    print(f"[GUI] Saved result image to {save_path}")

    # also pop up the system image viewer
    img.show()
    print("[GUI] Image viewer launched")




# ============================================================================
# PART 2: THE A* ALGORITHM (Simplified)
# ============================================================================

class Node:
    """A node in the A* search"""
    
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.g = float('inf')  # Cost from start to this node
        self.h = 0              # Heuristic (estimated cost to goal)
        self.f = float('inf')   # Total cost (g + h)
        self.parent = None      # Parent node for path reconstruction
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.f < other.f
    
    def __eq__(self, other):
        """For set membership"""
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        """For using nodes in sets and dictionaries"""
        return hash((self.x, self.y))
    
    def __repr__(self):
        """Printable representation, useful in debugging and logs"""
        return f"({self.x},{self.y})"

def astar_simple(grid: SimpleGrid, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    SIMPLIFIED A* ALGORITHM
    
    Args:
        grid: The grid world
        start: (x, y) start position
        goal: (x, y) goal position
    
    Returns:
        List of (x, y) positions from start to goal, or None if no path
    """
    
    # ------------------------------------------------------------------------
    # STEP 1: Initialize
    # ------------------------------------------------------------------------
    
    # Check if start and goal are valid
    if not grid.is_free(*start):
        print(f"Start position {start} is occupied!")
        return None
    if not grid.is_free(*goal):
        print(f"Goal position {goal} is occupied!")
        return None
    
    # Data structures:
    open_set = []          # Priority queue of (f, g, x, y) tuples
    closed_set = set()     # Set of already explored (x,y) positions
    all_nodes = {}         # (x,y) -> {'g': ..., 'parent': (px,py) or None}
    
    # Initialize start
    all_nodes[start] = {'g': 0, 'parent': None}
    f_start = heuristic_xy(start, goal)
    heapq.heappush(open_set, (f_start, 0.0, start[0], start[1]))
    
    # ------------------------------------------------------------------------
    # STEP 2: Define movement options (8 directions)
    # ------------------------------------------------------------------------
    
    # (dx, dy, cost) - 8-connected grid
    movements = [
        (1, 0, 1.0),    # right
        (-1, 0, 1.0),   # left
        (0, 1, 1.0),    # down
        (0, -1, 1.0),   # up
        (1, 1, 1.414),  # diagonal down-right
        (1, -1, 1.414), # diagonal up-right
        (-1, 1, 1.414), # diagonal down-left
        (-1, -1, 1.414) # diagonal up-left
    ]
    
    # ------------------------------------------------------------------------
    # STEP 3: Main A* loop
    # ------------------------------------------------------------------------
    
    iteration = 0
    print("\nStarting A* search...")
    
    while open_set:
        # Get node with lowest f-score from priority queue
        f_cur, neg, cx, cy = heapq.heappop(open_set)
        current = (cx, cy)
        
        # Skip if already expanded (lazy deletion)
        if current in closed_set:
            continue
        
        # Visualize progress every few iterations
        iteration += 1
        if iteration % 500 == 0:
            g_cur = all_nodes[current]['g']
            print(f"  Iteration {iteration}: Exploring {current}, "
                  f"f={f_cur:.2f}, g={g_cur:.2f}")
        
        # Check if we reached the goal
        if current == goal:
            print(f"✓ Goal reached after {iteration} iterations!")
            
            # Reconstruct path by following parent pointers
            path = []
            pos = current
            while pos is not None:
                path.append(pos)
                pos = all_nodes[pos]['parent']
            path.reverse()  # Reverse to get start -> goal
            
            # Compute actual path cost
            cost = all_nodes[current]['g']
            print(f"  Path cost: {cost:.2f}")
            return path
        
        # Mark current node as explored
        closed_set.add(current)
        g_cur = all_nodes[current]['g']
        
        # Explore all neighbors
        for dx, dy, move_cost in movements:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            
            # Skip if neighbor is not free or already explored
            if not grid.is_free(nx, ny):
                continue
            if neighbor in closed_set:
                continue
            
            # tentative_g = cost from start to current + cost to neighbor
            tentative_g = g_cur + move_cost
            
            # If this path to neighbor is better than any previous path
            old_g = all_nodes[neighbor]['g'] if neighbor in all_nodes else float('inf')
            if tentative_g < old_g:
                # Update / create node info
                all_nodes[neighbor] = {'g': tentative_g, 'parent': current}
                f_new = tentative_g + heuristic_xy(neighbor, goal)
                # Always push — stale entries are skipped when popped
                heapq.heappush(open_set, (f_new, tentative_g, nx, ny))
    
    # If we get here, open set is empty and goal not reached
    print(f"✗ No path found after {iteration} iterations!")
    return None


def heuristic_xy(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Heuristic function: Euclidean distance between two (x,y) positions.
    Admissible for 8-connected grids with diagonal cost sqrt(2).
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


# ============================================================================
# PART 3: DEMO - Run A* on a simple grid
# ============================================================================

def main():
    """Run a simple demo of A* path planning"""
    
    print("=" * 60)
    print("SIMPLIFIED A* PATH PLANNING DEMO")
    print("=" * 60)
    
    # If a map file exists in the `maps` directory we will use it instead
    # of generating a tiny toy grid.  This satisfies the user's request to
    # "implement this a star on the map_task1.pgm".
    map_file = "maps/map_task1.pgm"
    if os.path.exists(map_file):
        print(f"\nLoading occupancy map from {map_file}")
        grid = load_grid_from_pgm(map_file)
        # Start: leftmost free cell in the main connected region
        # Goal:  just to the right of the central obstacle
        start, _ = pick_endpoints(grid)
        goal = (240, 68)  # free area immediately right of central obstacle
        # safety: if the chosen goal is not free, nudge it
        if not grid.is_free(*goal):
            for dx in range(0, 20):
                for dy in range(-10, 11):
                    if grid.is_free(goal[0]+dx, goal[1]+dy):
                        goal = (goal[0]+dx, goal[1]+dy)
                        break
                else:
                    continue
                break
        print(f"Start (left side): {start}")
        print(f"Goal (right of central obstacle): {goal}")
    else:
        # fallback behaviour – original demo grid
        grid = SimpleGrid(15, 15)
        # Add some obstacles to make it interesting
        # Vertical wall
        for y in range(3, 12):
            grid.add_obstacle(7, y)
        # Horizontal wall with a gap
        for x in range(2, 13):
            if x != 7:  # Leave a gap at x=7
                grid.add_obstacle(x, 8)
        # Some random obstacles
        random.seed(42)  # For reproducible results
        grid.add_random_obstacles(10)
        # Set start and goal positions
        start = (2, 2)
        goal = (12, 12)
        print(f"\nGrid: {grid.width}x{grid.height}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Obstacles: {sum(row.count(1) for row in grid.grid)}")
    
    # Display the grid without path first
    print("\nInitial Grid:")
    grid.display()
    # also pop up a simple GUI image for the initial map
    try:
        display_gui(grid)
    except Exception as e:
        print(f"GUI display failed: {e}")
    
    # Run A*
    print("\n" + "=" * 60)
    print("RUNNING A* ALGORITHM")
    print("=" * 60)
    
    path = astar_simple(grid, start, goal)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if path:
        print(f"\n✓ Path found with {len(path)} steps!")
        print(f"Path: {path}")
        
        # Display grid with path
        print("\nGrid with path:")
        grid.display(path, start, goal)
        # show GUI with path overlay
        try:
            display_gui(grid, path, start, goal)
        except Exception as e:
            print(f"GUI display failed: {e}")
        
        # Optional: Print step-by-step directions
        print("\nStep-by-step directions:")
        for i, (x, y) in enumerate(path):
            if i == 0:
                print(f"  Step {i}: Start at ({x}, {y})")
            elif i == len(path) - 1:
                print(f"  Step {i}: Goal reached at ({x}, {y})")
            else:
                prev_x, prev_y = path[i-1]
                dx = x - prev_x
                dy = y - prev_y
                direction = ""
                if dx > 0 and dy == 0: direction = "right"
                elif dx < 0 and dy == 0: direction = "left"
                elif dy > 0 and dx == 0: direction = "down"
                elif dy < 0 and dx == 0: direction = "up"
                elif dx > 0 and dy > 0: direction = "down-right"
                elif dx > 0 and dy < 0: direction = "up-right"
                elif dx < 0 and dy > 0: direction = "down-left"
                elif dx < 0 and dy < 0: direction = "up-left"
                print(f"  Step {i}: Move {direction} to ({x}, {y})")
    else:
        print("\n✗ No path found!")
        grid.display(path, start, goal)
    
    print("\n" + "=" * 60)
    print("END OF DEMO")
    print("=" * 60)


# ============================================================================
# PART 4: UNDERSTANDING A* - Key Concepts Explained
# ============================================================================

"""
UNDERSTANDING A* ALGORITHM - KEY CONCEPTS:

1. g-score: Actual cost from start to current node
   - Like keeping track of how far you've traveled

2. h-score: Heuristic estimate from current node to goal
   - A "guess" of remaining distance (must be optimistic/underestimate)
   - Euclidean distance is used here (straight-line distance)

3. f-score: Total estimated cost (g + h)
   - This determines which node to explore next
   - Lower f-score = more promising path

4. Priority Queue (open_set):
   - Always expands the node with smallest f-score
   - Like saying "explore the most promising path first"

5. Closed Set:
   - Nodes already fully explored
   - Prevents revisiting nodes

6. Parent Pointers:
   - Each node remembers how we got there
   - Allows path reconstruction at the end

WHY A* WORKS:
- If h is admissible (never overestimates), A* guarantees optimal path
- It balances exploration (g) with goal-directedness (h)
- Much faster than Dijkstra (which only uses g)
- More reliable than greedy best-first (which only uses h)

TRADE-OFFS:
- Euclidean heuristic: More accurate but slower to compute
- Manhattan heuristic: Faster but less accurate for diagonal movement
- Diagonal heuristic: Good balance for 8-direction movement
"""

if __name__ == "__main__":
    main()
