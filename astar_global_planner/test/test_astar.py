"""Basic pytest stub for the astar_global_planner package."""

import pytest
from astar_global_planner.frontier_based_astar import (
    GridCostmap,
    astar_search,
    smooth_path,
)


def _make_free_grid(w: int = 20, h: int = 20, res: float = 1.0) -> GridCostmap:
    """Return a completely free grid."""
    g = GridCostmap()
    g.width = w
    g.height = h
    g.resolution = res
    g.origin_x = 0.0
    g.origin_y = 0.0
    g.data = [0] * (w * h)
    return g


def test_astar_finds_straight_path():
    grid = _make_free_grid()
    path = astar_search(grid, (0, 0), (5, 0), use_costmap_weights=False)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (5, 0)


def test_astar_finds_diagonal_path():
    grid = _make_free_grid()
    path = astar_search(grid, (0, 0), (5, 5), use_costmap_weights=False)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (5, 5)


def test_astar_blocked():
    grid = _make_free_grid(10, 10)
    # Wall across row 5
    for x in range(10):
        grid.data[5 * 10 + x] = 100
    path = astar_search(grid, (5, 0), (5, 9))
    assert path is None


def test_smooth_path_preserves_endpoints():
    raw = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 0.5), (4.0, 0.0)]
    s = smooth_path(raw)
    assert len(s) == len(raw)
    assert s[0] == raw[0]
    assert s[-1] == raw[-1]


def test_world_map_roundtrip():
    g = _make_free_grid(100, 100, res=0.05)
    g.origin_x = -2.5
    g.origin_y = -2.5
    mx, my = g.world_to_map(0.0, 0.0)
    wx, wy = g.map_to_world(mx, my)
    assert abs(wx - 0.0) < g.resolution
    assert abs(wy - 0.0) < g.resolution
