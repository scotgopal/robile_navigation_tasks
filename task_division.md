# AMR Project: Detailed Task Breakdown

This document outlines the specific duties for each of the three project tasks as defined in the course requirements.

---

## Task 1: Path and Motion Planning
**Objective:** Port simulation code to the real Robile and integrate global and local planning.

* **Member A (Lead)**
    * Port the existing potential field planner implementation to the physical Robile platform.
    * Integrate the A* path planner to find a rough global trajectory of waypoints.
    * Implement post-processing logic to extract representative waypoints (e.g., every $n$-th cell or sharp points) from the A* path.
* **Member B (Support)**
    * Validate the potential field planner's performance in avoiding real-world obstacles.
    * Assist in testing the transition between waypoints to ensure smooth movement.
* **Member C (Support)**
    * Provide the initial grid map generated via SLAM for use by the global planner.
    * Ensure the coordinate frames of the map are correctly interpreted by the planning nodes.

---

## Task 2: Localisation
**Objective:** Implement and integrate a custom Monte Carlo Localisation (MCL) particle filter.

* **Member B (Lead)**
    * Develop the core code for the simple particle filter version discussed in the lecture.
    * Integrate the custom filter onto the Robile system to handle real-time sensor data.
    * Optional: If time allows, explore extensions like the Adaptive Monte Carlo approach.
* **Member A (Support)**
    * Ensure the navigation system can subscribe to and utilize the pose estimates from the new particle filter.
* **Member C (Support)**
    * Maintain the environment map that serves as the basis for the localisation algorithm.

---

## Task 3: Environment Exploration
**Objective:** Combine a SLAM component with an automated exploration algorithm.

* **Member C (Lead - Your Role)**
    * Run a SLAM approach (like `slam_toolbox`) in parallel with the exploration logic.
    * Develop the algorithm to select poses at the **Kartenrand** (map fringe) between explored and unexplored regions.
    * Ensure pose selection is dynamically updated based on the most recent SLAM map.
* **Member A (Support)**
    * Connect the exploration component to the path planner so the robot can autonomously reach selected fringe poses.
* **Member B (Support)**
    * Monitor localisation stability while the map is being built and expanded by the SLAM algorithm.