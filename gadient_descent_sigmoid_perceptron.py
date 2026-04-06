import numpy as np

# Dataset
x1 = np.array([0, 1, 2, 2])
x2 = np.array([0, 0, 2, 3])
y  = np.array([0, 0, 1, 1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize weights
w0, w1, w2 = 0.0, 0.0, 0.0

# Hyperparameters
lr = 0.01
epochs = 2

# Training
for epoch in range(epochs):
    dw0, dw1, dw2 = 0.0, 0.0, 0.0
    for i in range(len(x1)):
        # Linear combination
        a = w0 + w1 * x1[i] + w2 * x2[i]
        
        # Prediction
        yo = sigmoid(a)
        
        # Error
        error = yo - y[i]
        
        dy = yo * (1 - yo)  
        
        # Gradients
        dw0 += error * dy
        dw1 += error * dy * x1[i]
        dw2 += error * dy * x2[i]
        
        print(f"w0 = {w0:.3f}, w1 = {w1:.3f}, w2 = {w2:.3f}")
        
    # Update weights
    w0 -= lr * dw0
    w1 -= lr * dw1
    w2 -= lr * dw2
    print(f"w0 = {w0:.3f}, w1 = {w1:.3f}, w2 = {w2:.3f}")    

# Final weights
print("\nFinal Weights:")
print(f"w0 = {w0:.3f}, w1 = {w1:.3f}, w2 = {w2:.3f}")