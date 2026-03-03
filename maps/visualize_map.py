#!/usr/bin/env python3
"""
Visualize a binary/occupancy map defined by a ROS-style map YAML file.

Reads the YAML (image path, resolution, origin, negate) and the referenced
PGM image, then displays it with matplotlib with axes in world coordinates (m).

Usage:
    python visualize_map.py [path_to_map.yaml]
    python visualize_map.py closed_walls_map.yaml
    python visualize_map.py map_task1.yaml -o map_task1.png   # save to file (no GUI)
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_map_yaml(yaml_path: Path) -> dict:
    """
    Parse map YAML and return dict with image, resolution, origin, negate.

    Parameters
    ----------
    yaml_path : pathlib.Path
        Path to the .yaml file.

    Returns
    -------
    dict
        Keys: image (str), resolution (float), origin ([x,y,yaw]), negate (int).
    """
    text = yaml_path.read_text()
    out = {
        "image": "map.pgm",
        "resolution": 0.05,
        "origin": [0.0, 0.0, 0.0],
        "negate": 0,
    }
    # image: closed_walls_map.pgm
    m = re.search(r"image:\s*(\S+)", text)
    if m:
        out["image"] = m.group(1).strip()
    # resolution: 0.05
    m = re.search(r"resolution:\s*([\d.]+)", text)
    if m:
        out["resolution"] = float(m.group(1))
    # origin: [-0.767, -4.85, 0]
    m = re.search(r"origin:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]", text)
    if m:
        out["origin"] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
    # negate: 0
    m = re.search(r"negate:\s*(\d+)", text)
    if m:
        out["negate"] = int(m.group(1))
    return out


def load_pgm(pgm_path: Path) -> np.ndarray:
    """
    Load a PGM image into a 2D numpy array (row, col), values in [0, 255].

    Parameters
    ----------
    pgm_path : pathlib.Path
        Path to the .pgm file.

    Returns
    -------
    np.ndarray
        Shape (height, width), dtype uint8.
    """
    with open(pgm_path, "rb") as f:
        # P5 binary PGM: "P5" then width height then maxval then raw bytes
        header = []
        while len(header) < 4:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("#"):
                continue
            header.extend(line.split())
        if header[0] != "P5":
            raise ValueError(f"Expected PGM P5, got {header[0]}")
        width, height = int(header[1]), int(header[2])
        maxval = int(header[3])
        data = np.frombuffer(f.read(), dtype=np.uint8 if maxval <= 255 else np.uint16)
    img = data.reshape((height, width))
    return img


def main():
    script_dir = Path(__file__).resolve().parent
    default_yaml = script_dir / "closed_walls_map.yaml"

    parser = argparse.ArgumentParser(description="Visualize a ROS-style occupancy map")
    parser.add_argument(
        "yaml",
        nargs="?",
        default=str(default_yaml),
        help="Path to map YAML (default: closed_walls_map.yaml in script dir)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default=None,
        help="Save figure to FILE instead of showing (e.g. for headless)",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml).resolve()
    if not yaml_path.is_file():
        raise SystemExit(f"Map YAML not found: {yaml_path}")

    meta = load_map_yaml(yaml_path)
    map_dir = yaml_path.parent
    pgm_path = map_dir / meta["image"]
    if not pgm_path.is_file():
        raise SystemExit(f"Map image not found: {pgm_path}")

    img = load_pgm(pgm_path)
    resolution = meta["resolution"]
    ox, oy, _ = meta["origin"]
    negate = meta["negate"]

    if negate:
        img = 255 - np.asarray(img, dtype=np.uint8)

    # World extent: x and y in meters (ROS map frame: x right, y up in image terms)
    # Image: first row is top (max y), first col is left (min x)
    height, width = img.shape
    x_min = ox
    x_max = ox + width * resolution
    y_min = oy
    y_max = oy + height * resolution

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(
        img,
        extent=[x_min, x_max, y_min, y_max],
        origin="upper",
        cmap="gray",
        interpolation="nearest",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(yaml_path.name)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
