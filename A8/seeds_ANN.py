#importing required libraries 
from email import header
from string import whitespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# activation function
def sigmoid(Z):
    ret = np.maximum(0, Z)
    return ret

#initializing random values to weights and bias of each layers of neurons
def param_init(layer_dims):
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*0.01
        parameters['B' + str(i)] = np.random.randn(layer_dims[i],1)*0.01
    return parameters

# forward propagation for calculating the activation values of neurons of next layers based on train dataset and parameters(W,B)
#formula used A = W₁.X + b₁
def forward_propagation(X_train, parameters): 
    layers = len(parameters)//2
    values = {}
    for i in range(1, layers+1):
        if i==1: #for first internal layer
            values['Z' + str(i)] = np.dot(parameters['W' + str(i)], X_train) + parameters['B' + str(i)]
            values['A' + str(i)] = sigmoid(values['Z' + str(i)])
        else: # for rest of the internal layers
            values['Z' + str(i)] = np.dot(parameters['W' + str(i)], values['A' + str(i-1)]) + parameters['B' + str(i)]
            if i==layers:
                values['A' + str(i)] = values['Z' + str(i)]
            else:
                values['A' + str(i)] = sigmoid(values['Z' + str(i)])
    return values

#  calculating gradient w.r.t weights and bias using weights, bias , activations and training data as input
def backward_propagation(parameters, values, X_train, Y_train):
    layers = len(parameters)//2
    m = len(Y_train)
    grads = {}
    for i in range(layers,0,-1):
        if i==layers:
            dA = 1/m * (values['A' + str(i)] - Y_train)
            dZ = dA
        else:
            dA = np.dot(parameters['W' + str(i+1)].T, dZ)
            dZ = np.multiply(dA, np.where(values['A' + str(i)]>=0, 1, 0))
        if i==1:
            grads['W' + str(i)] = 1/m * np.dot(dZ, X_train.T)
            grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        else:
            grads['W' + str(i)] = 1/m * np.dot(dZ,values['A' + str(i-1)].T)
            grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    return grads

# updating weights and bias using gradients and learning rates as input for decreasing cost and perform next cycle of forward-backward propagation
def param_updt(parameters, grads, learning_rate): 
    layers = len(parameters)//2
    parameters_updated = {}
    for i in range(1,layers+1):
        parameters_updated['W' + str(i)] = parameters['W' + str(i)] - learning_rate * grads['W' + str(i)]
        parameters_updated['B' + str(i)] = parameters['B' + str(i)] - learning_rate * grads['B' + str(i)]
    return parameters_updated

# model training function to train neural network based on no.of layers
def model(X_train, Y_train, layer_dims, num_iters, learning_rate): 
    parameters = param_init(layer_dims)
    for i in range(num_iters):
        values = forward_propagation(X_train.T, parameters)
        grads = backward_propagation(parameters, values,X_train.T, Y_train.T)
        parameters = param_updt(parameters, grads, learning_rate)
    return parameters

# based on updates valued of parameters, accuracy is calculated as RMSE
def accuracy(X_train, X_test, Y_train, Y_test, parameters): #compute accuracy on test and training data given learnt parameters
    values_train = forward_propagation(X_train.T, parameters)
    values_test = forward_propagation(X_test.T, parameters)
    train_acc = np.sqrt(mean_squared_error(Y_train, values_train['A' + str(len(layer_dims)-1)].T))
    test_acc = np.sqrt(mean_squared_error(Y_test, values_test['A' + str(len(layer_dims)-1)].T))
    return train_acc, test_acc

# importing the dataset
ds = pd.read_csv("seeds_dataset.txt",header=None, sep=' ')
# spliting into X and Y
print(ds.head())
X = ds.iloc[:, 0:6]
X = X.to_numpy()
y = ds.iloc[:, 6:7]
y =y.to_numpy()
print(y)
# Y = np.array([])
# for i in y:
#     Y = np.append(Y, i[0])
# # print(Y)

# # train-test split in 80:20 ratio
# print(X.shape, Y.shape)
# X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2,random_state=42)

# # considering 4 layered neural network with sizes given below
# layer_dims = [6, 4, 4, 3, 1]
# W_B = model(X_train,Y_train,layer_dims, 200, 0.05)

# train_accuracy, test_accuracy = accuracy(X_train, X_test, Y_train,Y_test, W_B)
# print(f'Accuracy of Train Dataset : {train_accuracy}')
# print(f'Accuracy of Test Dataset : {test_accuracy}')