#Q1 Simple linear Regression using scikit , dataset used : Salary_Data.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

# importing dataset
data_set = pd.read_csv("Salary_Data.csv")
ds = data_set.sample(frac=1)
print(f"a) Dataset Imported: \n {ds}")

# assigning Independent & dependent variables
X = ds.iloc[:,:-1].values
y = ds.iloc[:, 1].values
X = X.reshape(-1,1)
y = y.reshape(-1,1)

# splitting dataset into train and test in 80-20 %
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state = 0)

# plotting train Dataset
plt.scatter(X_train,y_train)
plt.show()

# plotting test Dataset
plt.scatter(X_test,y_test)
plt.show()

#fitting SLR using Train datset
slr = LinearRegression()
slr.fit(X_train,y_train)

# retreiving the slope,intercept
slope = slr.coef_
intercept = slr.intercept_
print(f"\nSlope and Intercept of regression line: {slope} {intercept}\n")

# predicting the values using test dataset
y_pred = slr.predict(X_test)
print(f"Predicted Values: {y_pred}")

# plotting scattered dataset and regression line 
plt.scatter(X_train,y_train, color='red',)
plt.plot(X_train,slr.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,y_test, color='red',)
plt.plot(X_train,slr.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#Calculate the error / accuracy of the model using root mean squared error
rmse = mse(y_test, y_pred, squared = False)
print("Root Mean Squared Error of the model is: ", rmse)