#Q1 Logistic Regression from scratch, dataset used : heart.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
data_set = pd.read_csv("heart.csv")
ds = data_set.sample(frac=1)
print(f"a) Dataset Imported: \n {ds}\n")

# spliting dataset into train and test 80:20 ratio
train_size = int(0.8*len(ds))
train_data = ds[:train_size]
test_data = ds.drop(train_data.index)
X_train = train_data[train_data.columns[:-1]]
Y_train = train_data[train_data.columns[-1]]
X_test = test_data[test_data.columns[:-1]]
Y_test = test_data[test_data.columns[-1]]
print(f"Train data: \n{train_data}\n")
print(f"Test data: \n{test_data}\n")

# normalizing train and test Dataset
X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()

# reshaping dataset
Y_train = Y_train.values.reshape(-1, 1)
Y_test = Y_test.values.reshape(-1, 1)
X_train = np.concatenate((np.ones(shape = Y_train.shape, dtype = np.float64), X_train), axis = 1)
X_test = np.concatenate((np.ones(shape = Y_test.shape, dtype = np.float64), X_test), axis = 1)

# hypothetical function
def h(X_train, theta):
    hyp = np.matmul(X_train, theta)
    g = 1/(1 + np.exp(-hyp))
    return g

# cost function
def cost(X_train, Y_train, theta):
    m = len(Y_train)
    g = h(X_train, theta)
    Y_t = np.transpose(Y_train)
    cost = -(1/m) * (Y_t @ np.log(g) + (1-Y_t) @ np.log(1-g))
    return cost

# gradient descent
X, Y = X_train, Y_train
X_t = np.transpose(X)
theta = np.zeros((X_train.shape[1], 1), dtype = np.float64)
alpha = 0.01
epochs = 1000

cost_list = []
m = X.shape[0]
n = X.shape[1]

# Gradient Descnet algorithm
for i in range(0, epochs):

    g = h(X, theta)
    pd = np.subtract(g, Y)
    pd = (X_t @ pd)
    theta = theta - alpha*(pd)

    # Finding the cost
    costt = cost(X, Y, theta)
    cost_list.append(costt[0][0])
    
#plotting cost vs epochs
plt.plot(list(range(0, epochs)), cost_list)
plt.title("Cost vs Epochs")
plt.show()

# final prediction and accuracy
def pred(X_test,Y_test, theta):
    prediction = h(X_test, theta)
    # predictions = np.where(predictions >= 0.5, 1, 0)
    for i in range(0,len(prediction)):
        if(prediction[i] >= 0.5): prediction[i] = 1
        else : prediction[i] = 0
    accuracy  = (prediction == Y_test).mean()
    return prediction,accuracy

prediction,accuracy = pred(X_test,Y_test,theta)
print(f"Accuracy :{accuracy}\n")

