"""
=============================================================================
Problem Description: UR5e Robotic Arm Grip Classification
=============================================================================

Scenario:
You are programming a UR5e robotic arm to sort parts in a factory. To 
determine whether a "gripping" operation is successful or failed, you 
collected sensor data and used two primary features for training your model:
1. Motor Torque (represented as x1)
2. Gripper Position Error (represented as x2)

Given Data:
After applying a Linear Support Vector Machine (SVM) algorithm, the model 
derived the following Decision Boundary equation:
    x1 + x2 - 4 = 0

From this equation, we can extract the weight vector W = [1, 1] and the 
constant bias b = 4. 

Additionally, the model identified a specific Support Vector representing 
a "Successful Grip" at the coordinates P(3, 2).

Objective:
Write a Python script to programmatically compute the following:
1. The perpendicular distance (d) between the support vector P and the 
   decision boundary.
2. The value of the SVM optimization objective function. Based on the 
   specific formulation in the lecture slides, the algorithm seeks to 
   maximize the margin by minimizing the function: (1/2) * ||W||.
=============================================================================
"""

import numpy as np

# 1. Define the given parameters from the problem
# Support Vector (P) for a successful grip operation
P = np.array([3, 2])

# Decision Boundary weights (W): x1 + x2 - 4 = 0
# W = [w1, w2]
W = np.array([1, 1])

# The constant 'b' based on the formula: wx - b = 0
b = 4 

# 2. Calculate the perpendicular distance (d) between the support vector and the decision boundary
# Formula: d = |w1*x1 + w2*x2 - b| / ||W||

# Calculate the numerator (absolute value of substituting the point into the line equation)
numerator = abs(np.dot(W, P) - b)

# Calculate the denominator (the Norm or magnitude of the weight vector W)
denominator = np.linalg.norm(W)

# Final calculated distance
d = numerator / denominator

# 3. Calculate the Optimization Objective value
# the goal is to minimize (1/2) * ||W||
optimization_value = 0.5 * np.linalg.norm(W)

# --- Print the results ---
print("=== UR5e Robotic Arm Problem Solution (SVM Optimization) ===")
print(f"Support Vector P: {P}")
print(f"Weight Vector W: {W}")
print(f"Constant b: {b}")
print("-" * 45)
print(f"1. Calculated Perpendicular Distance (d): {d:.4f}")
print(f"2. Weight Norm ||W||: {denominator:.4f}")
print(f"3. Optimization Objective Value (1/2 * ||W||): {optimization_value:.4f}")