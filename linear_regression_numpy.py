import numpy as np

X = [26,30,44,50,62,68,74]
Y = [92,85,78,80,54,51,40]

X = np.array(X)
Y = np.array(Y)
n = X.size
A = np.array([
    [n, np.sum(X)], 
    [np.sum(X), np.sum(X**2)]
   ] )
C = np.array([np.sum(Y),np.sum(X*Y)])
a0, a1 = np.linalg.solve(A,C)

print(a0, a1)

