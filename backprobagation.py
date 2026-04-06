import numpy as np

# Inputs
i1, i2 = 0.05, 0.10

# Targets
t1, t2 = 0.01, 0.99

# Initial weights
w1, w2 = 0.15, 0.20
w3, w4 = 0.25, 0.30
w5, w6 = 0.40, 0.45
w7, w8 = 0.50, 0.55

# Biases (not updated)
b1, b2 = 0.35, 0.60

lr = 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# ========== FORWARD PASS ==========

# Hidden layer
h1 = sigmoid(i1*w1 + i2*w2 + b1)
h2 = sigmoid(i1*w3 + i2*w4 + b1)

# Output layer
o1 = sigmoid(h1*w5 + h2*w6 + b2)
o2 = sigmoid(h1*w7 + h2*w8 + b2)

# ========== ERROR ==========
E = 0.5 * ((t1 - o1)**2 + (t2 - o2)**2)

# ========== BACKPROP ==========

# Output layer gradients
d_o1 = (o1 - t1) * sigmoid_deriv(o1)
d_o2 = (o2 - t2) * sigmoid_deriv(o2)

# Gradients for w5, w6, w7, w8
dw5 = d_o1 * h1
dw6 = d_o1 * h2
dw7 = d_o2 * h1
dw8 = d_o2 * h2

# Hidden layer gradients
d_h1 = (d_o1*w5 + d_o2*w7) * sigmoid_deriv(h1)
d_h2 = (d_o1*w6 + d_o2*w8) * sigmoid_deriv(h2)

# Gradients for w1, w2, w3, w4
dw1 = d_h1 * i1
dw2 = d_h1 * i2
dw3 = d_h2 * i1
dw4 = d_h2 * i2

# ========== UPDATE WEIGHTS ==========

w1 -= lr * dw1
w2 -= lr * dw2
w3 -= lr * dw3
w4 -= lr * dw4
w5 -= lr * dw5
w6 -= lr * dw6
w7 -= lr * dw7
w8 -= lr * dw8

# ========== PRINT RESULTS ==========

print("Updated weights (4 d.p):")
print(f"w1 = {w1:.4f}, w2 = {w2:.4f}")
print(f"w3 = {w3:.4f}, w4 = {w4:.4f}")
print(f"w5 = {w5:.4f}, w6 = {w6:.4f}")
print(f"w7 = {w7:.4f}, w8 = {w8:.4f}")

print(f"\nError = {E:.4f}")