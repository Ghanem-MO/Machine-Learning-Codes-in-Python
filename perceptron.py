## Implement the Perceptron learning algorithm in Python and run it with any example dataset.  

import numpy as np 

def unit_step(x):    
  return 1 if x >= 0 else 0 
  
def main():   
  # AND gate dataset  
  X = np.array([     
    [0, 0],        
    [0, 1],      
    [1, 0],         
    [1, 1]    
  ])    
  y = np.array([0, 0, 0, 1]) 
  
# Initialize weights and bias  
weights = np.zeros(2)   
bias = 0    
lr = 0.1   
epochs = 10   

# Training loop  
for _ in range(epochs):  
  for i in range(len(X)):         
    output = np.dot(X[i], weights) + bias       
    prediction = unit_step(output)        
    error = y[i] - prediction           
    weights += lr * error * X[i]        
    bias += lr * error 
