## Implement the Perceptron learning algorithm in Python and run it with any example dataset.  

import numpy as np 

def unit_step(a):    
  return 1 if a >= 0 else 0 
  
# AND gate dataset  
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])

y = np.array([0, 0, 0, 1]) 
  
# Initialize weights and bias  
w0 , w1 , w2 = 0 , 0 ,0     
lr = 0.1  
 
epochs = 10   

# Training loop  
for epoch in range(epochs):  
    for i in range(len(x1)):         
      a = w1*x1[i] + w2*x2[i] + w0       
      yo = unit_step(a)        
      error = y[i] - yo           
      w1 += lr * error * x1[i]       
      w2 += lr * error * x2[i]       
      w0 += lr * error * 1

