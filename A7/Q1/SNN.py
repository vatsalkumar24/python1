#importing reequired libraries
import pandas as pd                   
import numpy as np
import keras
from keras.models import Sequential   
from keras.layers import Dense  

# Load the dataset from the CSV file
ds = pd.read_csv('insurance_data.csv')
print(f'Datset: \n{ds.head()}')

#normalising Dataset except last column using Z-score
ds['Affordability'] = (ds['Affordability'] - ds['Affordability'].mean())/ds['Affordability'].std()
ds['age'] = (ds['age'] - ds['age'].mean())/ds['age'].std()

#spilitting dataset into train and test in 80:20
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

#initi8alising model using keras
model=Sequential()                                      
model.add(Dense(2, input_dim=2, activation='relu'))     
model.add(Dense(2, input_dim=2, activation='relu'))    
model.add(Dense(1, activation='softmax'))

#We use keras's compiler function to specify the loss and optimiizer.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training the model
model.fit(X_train, Y_train, epochs=100)

#Predicting the test data
model.predict(X_test)

#Finding the accuracy using evalute
results = model.evaluate(X_test, Y_test)

print('Test loss:', results[0]) 
print('Test accuracy:', results[1]*100)