# Logistic Regression for Multiclass classification
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#importing dataset
ds = pd.read_csv("Iris.csv")
ds["Species"].replace({"Iris-setosa": "0", "Iris-versicolor": "1","Iris-virginica": "2"}, inplace=True)
# print(ds)

#feature selection
features = ds.drop(["Species"],axis=1)
target_features = ds["Species"]
print("features: \n",features)
print("target features: \n",target_features)

#scalling/normalizing using min-max normalization
x = features.values.astype(float) 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scaled_features = pd.DataFrame(x_scaled)
# print(scaled_features)

#train-test split 80:20
train = scaled_features.sample(frac=0.8)
test = scaled_features.drop(train.index)
X_train = train
Y_train = target_features.sample(frac=0.8)
X_test = test
Y_test = target_features.drop(Y_train.index)



#creating arrays of values
X_train=X_train.values
Y_train=Y_train.values
X_test=X_test.values
Y_test=Y_test.values

#logistic regression implementation
slope=[]
intercept=[]
classs = ["0","1","2"]

for i in range(3):
    temp_m=[]
    for j in range(len(X_train[0])):
        temp_m.append(1)
    slope.append(temp_m)
    intercept.append(0)

alpha=0.01
n=len(X_train)

#gradient descent
epochs = 1000
for i in range(epochs):      
    for q in range((len(classs))):
    
        Y_curr=[]
        for j in range(n):
            if(Y_train[j]!=classs[q]):
                continue
            sum=0
            for k in range(5):
                sum+=(slope[q][k]*X_train[j][k])
            sum+=intercept[q]
            # Appending the sigmoid of the sum calculated
            Y_curr.append(1 / (1 + np.exp(-sum)))
    
        # Comparing Y_train predicted with the actual values of Y_train
        Y_curr=np.array(Y_curr).flatten()
        Y_dep=[]
        for p in range(len(Y_curr)):
            Y_dep.append(1)
        temp=Y_curr - Y_dep
    
        # Decreasing the value of C and m using gradient descent
        intercept[q]=intercept[q] - ((1/len(Y_curr))*alpha*(np.sum(temp)))
    
        for i in range(len(X_train[0])):
            sum=0
            cnt=0
            for j in range(n):
                if(Y_train[j]!=classs[q]):
                    continue
                sum+=(temp[cnt]*X_train[j][i])
                cnt+=1
            slope[q][i]=slope[q][i] - ((1/len(Y_curr))*alpha*sum)

print(f"Slope: {slope}\n")
print(f"Intercept: {intercept}\n")

#predicting using Y_test
Y_pred=[]
for i in range(len(X_test)):
    
    max=0
    for j in range(len(classs)):
        sum=0
        for k in range(len(X_test[0])):
            sum+=(slope[j][k]*X_test[i][k])
        sum+=intercept[j]
        val=1 / (1 + np.exp(-sum))
        
        if(max<val):
            max=val
            ind=j

    Y_pred.append(classs[ind])

#calculating the accuracy
cnt=0
correct=0
for i in range(np.size(Y_pred)):
  if(Y_test[i]==Y_pred[i]):
    correct+=1
  cnt+=1
accuracy=correct/cnt
print("Accuracy: ",accuracy*100)