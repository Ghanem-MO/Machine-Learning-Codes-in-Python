x = [2, 4, 6, 8, 10]
y = [11.5, 10.2,10.3,9.68,9.32]

def calculateError(x, y, w0, w1):
    e = 0
    for i in range(len(x)):
        e = e + (y[i] - (w0 + w1*x[i]))**2
    return e

def updateWeights(x, y, w0, w1, learningRate):
   de = 0
   for i in range(len(x)):
         de = de + (y[i] - (w0 + w1*x[i]))
    
   de = de * -2 / len(x)
   update_w0 = w0 - learningRate * de
   
   de = 0
   for i in range(len(x)):
         de = de + x[i] * (y[i] - (w0 + w1*x[i]))
   de = de * -2 / len(x)
   update_w1 = w1 - learningRate * de      
   
   return update_w0, update_w1  

wo = 0
w1 = 0
learningRate = 0.01

epochs = 1

for i in range(epochs):
    print("Epoch: ", i+1) 
    print("Initial error: ", calculateError(x, y, wo, w1))
    w0, w1 = updateWeights(x, y, wo, w1, learningRate)
    print("Updated weights: ", w0, w1)
    print("Error after update: ", calculateError(x, y, w0, w1))