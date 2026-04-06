X = [26,30,44,50,62,68,74]
Y = [92,85,78,80,54,51,40]

n = len(X)

print(n)

X2 = [x*x for x in X]
XY = [x*y for x,y in zip(X, Y)]
print(X2 , XY) 
sum_x = sum(X)
sum_x2 = sum(X2)
sum_y = sum(Y)
sum_xy = sum(XY)
det = n * sum_x2 - sum_x * sum_x
print(det)

a0 = (sum_x2 * sum_y - sum_x * sum_xy)/det
a1 = (-sum_x * sum_y +n * sum_xy)/det
print(a0, a1) 