#WAP to take a dataset of a system consist of following three instances applying linear regression with gradient descent take to minimize the cost function and perform the data visualization to demonstrate how the cost decreasing global minima plot the training data and regression plot the cost vs iteration
import numpy as np
import matplotlib.pyplot as plt
x=np.array([1,2,3],dtype=float)
y=np.array([2,3,4],dtype=float)
n=len(x)
m=0
b=0
rate=0.01
data=[]
iterations=1000
for i in range(iterations):
    y_pred=m*x+b
    error=y_pred-y
    cost=(1/(2*n))*np.sum(error**2)
    data.append(cost)
    dm=(1/n)*np.sum(error*x)
    db=(1/n)*np.sum(error)
    m=m-rate*dm
    b=b-rate*db
y_pred=m*x*b
print("Slope m:",m)
print("Intercept b: ",b)
print("Predicted value: ")
for i in range(n):
  print("x=",x[i],"Actual= ",y[i],"Predicted= ",y_pred[i])
  print("Final cost: ",data[-1])
  
plt.figure(figsize=(8,6))
plt.scatter(x,y,label="Training")
plt.plot(x,y_pred,label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(x,y,label="Training")
plt.plot(x,y_pred,label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()    
