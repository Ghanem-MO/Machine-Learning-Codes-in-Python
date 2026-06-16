import numpy as np

# ==========================================
# 1. Initialization
# ==========================================
# Inputs & Targets
x1, x2 = 0.05, 0.10
d1, d2 = 0.01, 0.99
eta = 0.5  # Learning rate

# Weights & Biases
w1, w2, w3, w4 = 0.15, 0.25, 0.20, 0.30
w5, w6 = 0.35, 0.35  # Biases for hidden layer
w7, w8, w9, w10 = 0.40, 0.50, 0.45, 0.55
w11, w12 = 0.60, 0.60  # Biases for output layer

# Activation Function & Derivative
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def sigmoid_deriv(y):
    # Derivative based on output: phi'(a) = phi(a)(1 - phi(a))
    return y * (1 - y)

# ==========================================
# 2. Forward Path
# ==========================================
# print("--- Forward Path ---")

a1 = (x1 * w1) + (x2 * w3) + w5
h1 = sigmoid(a1)
phi_prime_a1 = sigmoid_deriv(h1)

a2 = (x1 * w2) + (x2 * w4) + w6
h2 = sigmoid(a2)
phi_prime_a2 = sigmoid_deriv(h2)

a3 = (h1 * w7) + (h2 * w9) + w11
y1 = sigmoid(a3)
phi_prime_a3 = sigmoid_deriv(y1)

a4 = (h1 * w8) + (h2 * w10) + w12
y2 = sigmoid(a4)
phi_prime_a4 = sigmoid_deriv(y2)

# print(f"a1 = {a1:.4f}, y1 = {y1:.4f}, phi'(a1) = {phi_prime_a1:.4f}")
# print(f"a2 = {a2:.4f}, y2 = {y2:.4f}, phi'(a2) = {phi_prime_a2:.4f}")
# print(f"a3 = {a3:.4f}, y3 = {y3:.4f}, phi'(a3) = {phi_prime_a3:.4f}")
# print(f"a4 = {a4:.4f}, y4 = {y4:.4f}, phi'(a4) = {phi_prime_a4:.4f}\n")

# ==========================================
# 3. Backward Path
# ==========================================
# print("--- Backward Path ---")

# Error calculation
e1 = d1 - y1
e2 = d2 - y2
E = 0.5 * ((e1**2) + (e2**2))

# print(f"e1 = {e1:.4f}, e2 = {e2:.4f}")
print(f"Total Error (E) = {E:.4f}\n")

# print("--- Gradients (Output Layer) ---")
# Gradients for Output Layer Weights
dE_dw7  = -e1 * phi_prime_a3 * h1
dE_dw8  = -e2 * phi_prime_a4 * h1
dE_dw9  = -e1 * phi_prime_a3 * h2
dE_dw10 = -e2 * phi_prime_a4 * h2
dE_dw11 = -e1 * phi_prime_a3 * 1.0  # Bias
dE_dw12 = -e2 * phi_prime_a4 * 1.0  # Bias

# print(f"dE/dw7  = {dE_dw7:.4f}")
# print(f"dE/dw8  = {dE_dw8:.4f}")
# print(f"dE/dw9  = {dE_dw9:.4f}")
# print(f"dE/dw10 = {dE_dw10:.4f}")
# print(f"dE/dw11 = {dE_dw11:.4f}")
# print(f"dE/dw12 = {dE_dw12:.4f}\n")

# print("--- Propagate Error (Hidden Layer Deltas) ---")
# Delta calculations
delta1 = (-e1 * phi_prime_a3 * w7) + (-e2 * phi_prime_a4 * w8)
delta2 = (-e1 * phi_prime_a3 * w9) + (-e2 * phi_prime_a4 * w10)

print(f"delta1 (d1) = {delta1:.4f}")
print(f"delta2 (d2) = {delta2:.4f}\n")

# print("--- Gradients (Hidden Layer) ---")
# Gradients for Hidden Layer Weights
dE_dw1 = delta1 * phi_prime_a1 * x1
dE_dw2 = delta2 * phi_prime_a2 * x1
dE_dw3 = delta1 * phi_prime_a1 * x2
dE_dw4 = delta2 * phi_prime_a2 * x2
dE_dw5 = delta1 * phi_prime_a1 * 1.0  # Bias
dE_dw6 = delta2 * phi_prime_a2 * 1.0  # Bias

# print(f"dE/dw1 = {dE_dw1:.3e}")
# print(f"dE/dw2 = {dE_dw2:.3e}")
# print(f"dE/dw3 = {dE_dw3:.3e}")
# print(f"dE/dw4 = {dE_dw4:.3e}")
# print(f"dE/dw5 = {dE_dw5:.3e}")
# print(f"dE/dw6 = {dE_dw6:.3e}\n")

# ==========================================
# 4. Weight Updates
# ==========================================
# print("--- Final Updated Weights ---")
w1_new = w1 - (eta * dE_dw1)
w2_new = w2 - (eta * dE_dw2)
w3_new = w3 - (eta * dE_dw3)
w4_new = w4 - (eta * dE_dw4)
w5_new = w5 - (eta * dE_dw5)
w6_new = w6 - (eta * dE_dw6)

w7_new  = w7  - (eta * dE_dw7)
w8_new  = w8  - (eta * dE_dw8)
w9_new  = w9  - (eta * dE_dw9)
w10_new = w10 - (eta * dE_dw10)
w11_new = w11 - (eta * dE_dw11)
w12_new = w12 - (eta * dE_dw12)

print(f"w1  = {w1_new:.4f}, w2  = {w2_new:.4f}")
print(f"w3  = {w3_new:.4f}, w4  = {w4_new:.4f}")
print(f"w5  = {w5_new:.4f}, w6  = {w6_new:.4f}")
print(f"w7  = {w7_new:.4f}, w8  = {w8_new:.4f}")
print(f"w9  = {w9_new:.4f}, w10 = {w10_new:.4f}")
print(f"w11 = {w11_new:.4f}, w12 = {w12_new:.4f}")
