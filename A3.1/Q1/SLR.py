#Q1 Simple linear Regression frrom scratch , dataset used : Salary_Data.csv

#importing neccessary libraries
import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# a. importing dataset
data_set = pd.read_csv("Salary_Data.csv")
ds = data_set.sample(frac=1)
print(f"a) Dataset Imported: \n {ds}")

# b. assigning values to independent(X) & dependent(Y) varibales
X = ds['YearsExperience'].values
Y = ds['Salary'].values
print(f"b) Independent Variable: YearsExperience\n{X}\n")
print(f"b) Dependent Variable: Salary\n{Y}\n")

# c. Printing first few rows using head()
print("c. First few rows: ")
print(ds.head(),"\n")

# d. Separate the dataset into train and test data as 80% and 20% respectively.
train_size = int(0.8*len(ds))
train_data = ds[:train_size]
test_data = ds[train_size:]
X_train = X[:train_size]
X_test = X[train_size:]
Y_train = Y[:train_size]
Y_test = Y[train_size:]
print(f"d. Train Dataset: \n{train_data}\n")
print(f"d. Test Dataset: \n{test_data}\n")

# e. Plot the dataset (train dataset)( independent variable vs dependent variable)
#plotting train dataset
train_data.plot(x='YearsExperience',y='Salary', kind = 'scatter')
#displaying train Dataset
print("e. Ploting train dataset: \n")
plt.title("Train Dataset")
plt.show()

# f. Calculate the regression line(train dataset): i. Compute the slope ii. Compute the intercept iii. Compute and Plot regression line with data points
def linear_regression(x, y):     
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = ((x - x_mean)**2).sum()
    slope = num / den
    
    intercept = y_mean - (slope*x_mean)
    
    reg_line = 'y = {} + {}x'.format(intercept, round(slope, 3))
    
    return (intercept, slope, reg_line)

intercept,slope,reg_line = linear_regression(X_train,Y_train)
print("f. Calculate the regression line(train dataset):\n")
print(f"(i) Slope: {slope}")
print(f"(ii) Intercept: {intercept}")
print(f"(iii) Regression Line: {reg_line}\n")
# first ploting the dataset as scattered
train_data.plot(x='YearsExperience',y='Salary',kind='scatter')
# now plotting the reg_line obtained above
plt.plot(X_train, slope*X_train + intercept)
plt.title("Regression line with data poitns")
plt.show()


# g. Predict the values using test data.
prediction = slope*X_test + intercept
print(f"g. Predicted value using test data: {prediction}\n")

# h. Calculate the error / accuracy of the model using root mean squared error
error = math.sqrt(np.mean((Y_test - prediction)**2))
print(f"h.Root Mean Squared Error: {error}")