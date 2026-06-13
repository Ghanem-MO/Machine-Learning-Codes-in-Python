"""
=============================================================================
Problem Description: Obstacle Clustering in Robot Maps (K-Means Loop)
=============================================================================

Scenario:
During an autonomous exploration mission, a Pioneer 3-DX robot uses its LIDAR 
sensor to capture the (x, y) coordinates of 4 obstacles in an unknown environment. 
The system needs to cluster these obstacles into two groups (K=2).

Given Data:
- Obstacles: O1(2.0, 2.0), O2(3.0, 2.0), O3(8.0, 7.0), O4(9.0, 8.0)
- Initial Centroids: C1(2.0, 2.0), C2(9.0, 8.0)

Objective:
Write a Python script that repeats the K-Means steps (Assignment and Update) 
in a loop until convergence (the centroids stop changing).
=============================================================================
"""

import numpy as np

# 1. Coordinates of the obstacles detected by the Pioneer 3-DX robot
obstacles = np.array([
    [2.0, 2.0],  # O1
    [3.0, 2.0],  # O2
    [8.0, 7.0],  # O3
    [9.0, 8.0]   # O4
])

# 2. Initial Centroids
centroids = np.array([
    [2.0, 2.0],  # C1
    [9.0, 8.0]   # C2
])

iteration = 1

# Start the repeating loop
while True:
    print(f"\n========== Iteration {iteration} ==========")
    
    # Store old centroids to check for convergence later
    old_centroids = np.copy(centroids)
    
    # Lists to store the points assigned to each cluster for this iteration
    cluster_1_points = []
    cluster_2_points = []
    
    # --- Phase 1: Assignment Step ---
    print("--- Phase 1: Assignment Step ---")
    for i, point in enumerate(obstacles):
        # Calculate the Euclidean distance to each centroid
        dist_to_c1 = np.linalg.norm(point - centroids[0])
        dist_to_c2 = np.linalg.norm(point - centroids[1])
        
        # Assign the point to the closest centroid
        if dist_to_c1 < dist_to_c2:
            cluster_1_points.append(point)
            print(f"Obstacle O{i+1} {point} -> Assigned to C1 (Distance: {dist_to_c1:.2f})")
        else:
            cluster_2_points.append(point)
            print(f"Obstacle O{i+1} {point} -> Assigned to C2 (Distance: {dist_to_c2:.2f})")

    # --- Phase 2: Update Step ---
    print("--- Phase 2: Update Step ---")
    if cluster_1_points:
        centroids[0] = np.mean(np.array(cluster_1_points), axis=0)
        print(f"Updated Centroid C1: {centroids[0]}")
        
    if cluster_2_points:
        centroids[1] = np.mean(np.array(cluster_2_points), axis=0)
        print(f"Updated Centroid C2: {centroids[1]}")
        
    # --- Check for Convergence ---
    # If the centroids are exactly the same as the old centroids, the algorithm is finished
    if np.array_equal(centroids, old_centroids):
        print("\n*** CONVERGENCE REACHED ***")
        print("The centroids did not change. The K-Means algorithm is complete.")
        break
        
    iteration += 1
    
print(f"Number Of Iteration is " ,iteration)    
print("\n=== Final Results ===")
print(f"Final Danger Zone 1 (C1): {centroids[0]}")
print(f"Final Danger Zone 2 (C2): {centroids[1]}")