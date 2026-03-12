# Saving /map for unit tests

With your simulation (or map server) running, use either method below.

## Option 1: Python script (recommended – compact .npz for tests)

```bash
cd robile_navigation_tasks/localization && python3 scripts/save_map_for_tests.py
```

Writes `tests/fixtures/map_sample.npz`. Unit tests can load it with:

```python
data = np.load("tests/fixtures/map_sample.npz")
grid = data["grid"]
origin_x, origin_y = float(data["origin_x"]), float(data["origin_y"])
resolution = float(data["resolution"])
```

## Option 2: Bash one-liner (raw YAML)

```bash
ros2 topic echo /map --once > map_sample.yaml
```

Saves one message to `map_sample.yaml` in the current directory. The file can be large for big maps. Parsing in Python would require PyYAML and rebuilding the grid from the message fields.
