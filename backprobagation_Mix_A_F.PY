import numpy as np

# =====================================================================
# 1. ACTIVATION FUNCTIONS & DERIVATIVES
# Note: The derivatives accept the ALREADY ACTIVATED output.
# =====================================================================

# --- ReLU ---
# Law: f(x) = max(0, x)
def relu(x): return np.maximum(0, x)
def relu_deriv(out): return (out > 0).astype(float)

# --- Sigmoid ---
# Law: f(x) = 1 / (1 + e^(-x))
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(out): return out * (1 - out)

# --- Tanh ---
# Law: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
def tanh(x): return np.tanh(x)
def tanh_deriv(out): return 1.0 - out**2

# --- Linear ---
# Law: f(x) = x
def linear(x): return x
def linear_deriv(out): return 1.0

# --- Softmax (Utility only for this specific script) ---
# Law: f(x_i) = e^(x_i) / Sum(e^(x_j)) for all j
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# --- Softplus ---
# Law: f(x) = ln(1 + e^x)
def softplus(x): return np.log1p(np.exp(x))
def softplus_deriv(out): return 1.0 - np.exp(-out)

# --- Step Function (Threshold / Heaviside) ---
# Law: f(x) = 1 if x >= 0, else 0
def step(x): 
    return np.where(x >= 0, 1.0, 0.0)
def step_deriv(out): 
    # Derivative is 0 everywhere. Backprop will stall.
    return 0.0 * out 

# --- Signum Function (Bipolar) ---
# Law: f(x) = 1 if x >= 0, else -1
def signum(x): 
    return np.where(x >= 0, 1.0, -1.0)
def signum_deriv(out): 
    # Derivative is 0 everywhere. Backprop will stall.
    return 0.0 * out 

# --- Piecewise Linear Function ---
# Law: f(x) = mx + b (where the result is clamped between 0 and 1)
# (Using m=0.125, b=0.5 to approximate the graph in the slide)
def piecewise_linear(x, m=0.125, b=0.5):
    y = (m * x) + b
    return np.clip(y, 0.0, 1.0)
def piecewise_linear_deriv(out, m=0.125):
    # Derivative is 'm' only if the output is not capped at 0 or 1.
    return np.where((out > 0) & (out < 1), m, 0.0)


# =====================================================================
# 2. CHOOSE YOUR ACTIVATION FUNCTIONS HERE
# =====================================================================

# Currently set to Sigmoid for both to match your handwritten example!
hidden_activation = sigmoid
hidden_deriv = sigmoid_deriv

output_activation = sigmoid
output_deriv = sigmoid_deriv

# ==========================================
# 3. INITIALIZATION (From your specific example)
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


# ==========================================
# 4. FORWARD PATH
# ==========================================
# print("--- Forward Path ---")

a1 = (x1 * w1) + (x2 * w3) + w5
y1 = hidden_activation(a1)
phi_prime_a1 = hidden_deriv(y1)

a2 = (x1 * w2) + (x2 * w4) + w6
y2 = hidden_activation(a2)
phi_prime_a2 = hidden_deriv(y2)

a3 = (y1 * w7) + (y2 * w9) + w11
y3 = output_activation(a3)
phi_prime_a3 = output_deriv(y3)

a4 = (y1 * w8) + (y2 * w10) + w12
y4 = output_activation(a4)
phi_prime_a4 = output_deriv(y4)

# print(f"a1 = {a1:.4f}, y1 = {y1:.4f}, phi'(a1) = {phi_prime_a1:.4f}")
# print(f"a2 = {a2:.4f}, y2 = {y2:.4f}, phi'(a2) = {phi_prime_a2:.4f}")
# print(f"a3 = {a3:.4f}, y3 = {y3:.4f}, phi'(a3) = {phi_prime_a3:.4f}")
# print(f"a4 = {a4:.4f}, y4 = {y4:.4f}, phi'(a4) = {phi_prime_a4:.4f}\n")

# ==========================================
# 5. BACKWARD PATH
# ==========================================
# print("--- Backward Path ---")

# Error calculation
e1 = d1 - y3
e2 = d2 - y4
E = 0.5 * ((e1**2) + (e2**2))

# print(f"e1 = {e1:.4f}, e2 = {e2:.4f}")
print(f"Total Error (E) = {E:.4f}\n")

# print("--- Gradients (Output Layer) ---")
# Gradients for Output Layer Weights
dE_dw7  = -e1 * phi_prime_a3 * y1
dE_dw8  = -e2 * phi_prime_a4 * y1
dE_dw9  = -e1 * phi_prime_a3 * y2
dE_dw10 = -e2 * phi_prime_a4 * y2
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
# 6. WEIGHT UPDATES
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

print("Updated weights:")
print(f"w1  = {w1_new:.4f}, w2  = {w2_new:.4f}")
print(f"w3  = {w3_new:.4f}, w4  = {w4_new:.4f}")
print(f"w5  = {w5_new:.4f}, w6  = {w6_new:.4f}")
print(f"w7  = {w7_new:.4f}, w8  = {w8_new:.4f}")
print(f"w9  = {w9_new:.4f}, w10 = {w10_new:.4f}")
print(f"w11 = {w11_new:.4f}, w12 = {w12_new:.4f}")