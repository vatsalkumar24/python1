# Q2 Multi LInear Regression from scratch, dataset used: BOSTON HOUSING

#importing neccessary libraries
import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# a. importing dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data_set = pd.read_csv("housing.csv",header=None, delimiter=r"\s+", names = column_names)
ds = data_set
print("a. Imported Dataset: (first few rows)\n",ds.head())
print("\n")

#b. Scale the dataset [Standardization of the data]
#formulae is standardized_data = standardized_value = (value - mean) / std_deviation
for c in column_names:
     ds[c] = (ds[c] - ds[c].mean()) / ds[c].std()
print(f"b. Scaled Dataset using Standardization: \n{ds.head()} \n")

#c. Separate the dataset into train and test data as 80% and 20% respectively.
train_size = int(0.8*len(ds))
train_data = ds[:train_size] 
r = train_data.shape[0]
c = train_data.shape[1]
test_data = ds[train_size:]
X_train = train_data.iloc[:,:c-1].values
X_test = test_data.iloc[:,:c-1].values
Y_train = train_data.iloc[:,:c-1:c].values
Y_test = test_data.iloc[:,:c-1].values
print(f"c. Train Dataset: \n{train_data}\n")
print(f"c. Test Dataset: \n{test_data}\n")

#d. Implement the logic of the algorithm using Gradient Descent Function.
def CostFunction(x,y,w,b):
    cost = np.sum((((x.dot(w) + b) - y) ** 2) / (2*len(y)))
    return cost

def Gradient_Descent(x, y, w, b, learning_rate, epochs):
    cost_list = [0] * epochs
   
    for epoch in range(epochs):
        z = x.dot(w) + b
        loss = z - y
        
        weight_gradient = x.T.dot(loss) / len(y)
        bias_gradient = np.sum(loss) / len(y)
        
        w = w - learning_rate*weight_gradient
        b = b - learning_rate*bias_gradient
  
        cost = CostFunction(x, y, w, b)
        cost_list[epoch] = cost
               
    return w, b, cost_list

w, b, c= Gradient_Descent(X_train, Y_train, np.ones((X_train.shape[1],1),dtype=np.float64), 0, 0.05,epochs=10000)

print(f"d. Gradient Descent(w,b): {w,b}\n")

#e.Train the model and plot the data
plt.plot(c)
plt.show()

#f. Predict the values using test data.
predicted = X_test @ np.ones((X_train.shape[1], 1), dtype = np.float64)
print(f"f. predicted values using test data: ")
for i in range(0,len(predicted)):
    print(predicted[i],end=" ")

#g. Calculate the error / accuracy of the model using root mean squared error
error = np.sqrt(np.mean((Y_test - predicted)**2))
print(f"\n\ng. Root mean squared error: {error}")
