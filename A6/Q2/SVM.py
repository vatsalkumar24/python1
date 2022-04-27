#SVM for linear and polynomial kernel from scratch
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#importing dataset
dataset = pd.read_csv("Iris.csv")
ds=dataset.drop('Id',axis=1)

#creating numpy array
ds = np.array(ds)

#creating dependent and independent variables
X=ds[:100,:-1]
Y=ds[:100,-1]
y_unique=np.unique(Y)
for i in range (len(Y)):
  Y[i]=list(y_unique).index(Y[i])
Y=Y.astype('int32')
Y=np.array(Y)
Y=np.where(Y<=0,-1,1)
X=preprocessing.scale(X)

#train-test split
X_train, X_test, Y_train,Y_test = train_test_split(X,Y, test_size = 0.2,random_state=42)

#training and predicting functions
def train(X,y,epochs,lr,l):
  m,n=X.shape
  w=np.zeros(n)
  b=0
  for _ in range(epochs):
    for i,x in enumerate(X):
      condition=y[i]*(np.dot(x,w)-b)>=1
      if condition:
        w-=lr*(2*l*w)
      else:
        w-=lr*(2*l*w-np.dot(x,y[i]))
        b=lr*y[i]
  return w,b

def predict(X,w,b):
  y_pred=np.dot(X,w)-b
  return np.sign(y_pred)

#training and testing
w,b=train(X_train,Y_train,1000,0.5,0.001)
print(w," ",b)
y_pred=predict(X_test,w,b)
print(y_pred)
print(Y_test)

