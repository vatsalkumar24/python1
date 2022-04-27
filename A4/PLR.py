# Implementing Polinomial Regresssion from scratch usinf Position_Salaries dataset
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Datset
ds = pd.read_csv('Position_Salaries.csv')
print(f"Dataset: \n{ds}")

#droping position column as it is not needed of training dataset
ds = ds.drop(['Position'], axis=1)

#assign Independent(Level) and Dependent(Salary) variables
X = ds.iloc[:,:1]
y = ds.iloc[:,1:2]
X = X.astype('float32')
y = y.astype('float32')
print(f" X variable(Level): \n{X}")
print(f" Y variable(Salary): \n{y}")

#generating polynomial data
def get_polynomial(x, degree):
    data = pd.DataFrame(np.ones((x.shape[0], degree+1)))
    for i in range(1,degree+1):
        data.iloc[:,i] = (x**i).iloc[:,0]
    return data

def predict(X, theta):
    return np.matmul(X, theta)

def cost(X, y, theta):
    y_pred = predict(X, theta)
    error = y_pred - y
    cost = (1/2*y.shape[0])*np.dot(error.T,error)
    return cost

def gradient_descent(X, y, lr=0.01, num_epochs=100):
    m,n = X.shape
    theta = np.zeros([n,1])
    for epoch in range(num_epochs):
        y_pred = predict(X, theta)
        error = y_pred - y
        grad_theta = (1/m)*np.matmul(X.T, error)
        theta -= lr * grad_theta
    return theta

# Training the model
num_epochs = 10000
lr = 0.01
# Generate polynomial regression:
for i in range(1, 10):
    x = get_polynomial(X, i)
    x = x/x.max()
    x = x.values
    Y = y.values
    theta = gradient_descent(x, Y, lr, num_epochs)
    y_pred = predict(x, theta)
    plt.scatter(x[:,1],Y)
    plt.plot(x[:,1], y_pred, 'g')
    plt.title(f'Degree of Polynomial : {i}')
    plt.xlabel('Level')
    plt.ylabel('Salary')
    plt.show()