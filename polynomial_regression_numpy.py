import numpy as np

# Data
x = np.array([0,1,2,3,4,5])
y = np.array([2.1,7.7,13.6,27.2,40.9,61.1])

n = len(x)

# Sums
sx = np.sum(x)
sx2 = np.sum(x**2)
sx3 = np.sum(x**3)
sx4 = np.sum(x**4)

sy = np.sum(y)
sxy = np.sum(x*y)
sx2y = np.sum((x**2)*y)

# Matrix A
A = np.array([
    [n, sx, sx2],
    [sx, sx2, sx3],
    [sx2, sx3, sx4]
])

# Matrix C
C = np.array([sy, sxy, sx2y])

# Solve for coefficients
a0, a1, a2 = np.linalg.solve(A, C)

print("a0 =", a0)
print("a1 =", a1)
print("a2 =", a2)

print("\nPolynomial equation:")
print("y =", a0, "+", a1, "*x +", a2, "*x^2")