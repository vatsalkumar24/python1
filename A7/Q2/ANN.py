# importing all required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from sklearn.metrics import accuracy_score 
from keras.models import Sequential   
from keras.layers import Dense        

#loading the dataset
ds = pd.read_csv('Churn_Modelling.csv')
print(f'Datset: \n{ds.head()}')

#droping unwanted columns
ds.drop(ds.columns[[0,1,2]],axis=1,inplace=True)

#Encoding categorical features which are gender & Geography.
geo = ds.iloc[:,1].unique()
geo_encoded = {}
for i in range(len(geo)):
    geo_encoded[geo[i]] = i

gender = ds.iloc[:, 2].unique()
gen_encoded = {}
for i in range(len(gender)):
    geo_encoded[gender[i]] = i

# replacing with encoded values
ds["Geography"] = ds["Geography"].map(geo_encoded)
ds["Gender"] = ds["Gender"].map(gen_encoded)

#spliting into train test in 80:20 ratio
train = ds.sample(frac=0.8)
test = ds.drop(train.index)
print(f'Train Dataset: \n{train} \n')
print(f'Test Dataset: \n{test}\n')

#spilting train and test dataset into X and Y
X_train = train[train.columns[:-1]]
X_train = X_train.values
Y_train = train[train.columns[-1]]
Y_train = Y_train.values
X_test = test[test.columns[:-1]]
X_test = X_test.values
Y_test = test[test.columns[-1]]
Y_test = Y_test.values


# normalising Dataset using sklearn standardscalar
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.fit_transform(X_test) 

#initi8alising model using keras
model=Sequential()                                      
model.add(Dense(10, input_dim=10, activation='relu'))     
model.add(Dense(8, activation='relu'))      
model.add(Dense(6, activation='relu'))     
model.add(Dense(1, activation='sigmoid'))  

#We use keras's compiler function to specify the loss and optimiizer.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training the model
model.fit(X_train, Y_train, epochs=100)

#Predicting the test data
pred = model.predict(X_test)
for i in range(len(pred)):
    if pred[i][0] >= 0.5:
        pred[i] = 1
    else:
        pred[i] = 0

#Finding the accuracy using evalute
results = model.evaluate(X_test, Y_test)
# acc = accuracy_score(pred,Y_test.tolist())  
print('Test Accuracy: ', results[1]*100) 
