# Frame validation for MCL

## Run the script (with sim running)

```bash
cd robile_navigation_tasks/localization && python3 scripts/validate_frames.py
```

This prints:
- **/scan**: `header.frame_id`, number of beams, angle_min/increment, range_min/max
- **/map**: `header.frame_id`, origin (x,y), resolution, width×height
- **TF** `base_footprint` → scan frame: translation (laser origin in base); or a failure message
- A short **checklist** you can answer to confirm frames

---

## Questions you can validate yourself

1. **Is the map the same as the simulation?**  
   The map used by MCL (e.g. from `/map`) must be the same environment and frame as the sim. If the map was built from another run or has a different origin/resolution, localization will not converge.

2. **What is the scan’s `frame_id`?**  
   - Run: `ros2 topic echo /scan --once` and check `header.frame_id`.  
   - If it is `base_footprint`, laser origin is (0,0).  
   - If it is something else (e.g. `base_laser_front_link`), either:
     - TF `base_footprint` → that frame must be available, or  
     - Set `laser_origin_x` and `laser_origin_y` in `mcl_params.yaml` to the laser position in the base frame (e.g. from your URDF).

3. **Can TF resolve base → scan frame?**  
   The script reports this. If it fails, set `laser_origin_x` and `laser_origin_y` manually.

4. **When you set “2D Pose Estimate” in RViz, are you clicking where the robot actually is?**  
   Large initial error can prevent the particle cloud from converging.

5. **Map frame and origin**  
   - Run: `ros2 topic echo /map --once` and check `info.origin.position` and `info.resolution`.  
   - Compare with the script output; they should match what the sim uses.
