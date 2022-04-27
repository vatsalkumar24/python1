# PCA from Scratch
#importing required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# a. importing Dataset: hear
ds = pd.read_csv('framingham.csv')
ds = ds.dropna()
print(f"a. Dataset: \n {ds.head()}\n")

# b. Scale the Dataset 
Y = ds['TenYearCHD']
X = ds.drop(columns = ['TenYearCHD'])
X_scaled = StandardScaler().fit_transform(X)
print(f"b. Scaled Dataset: {X_scaled}\n")
# X_scaled = X_scaled.values


# d. Select the principal components

def PCA(X , k):
    # Subtract the mean of each of the features
    X = X - np.mean(X , axis = 0)
     
    # Calculate the covariance matrix of all features
    covX = np.cov(X , rowvar = False)
     
    # Calculate the eigen values and vectors
    eigenValues , eigenVectors = np.linalg.eigh(covX)
     
    # Sort the vectors according to their eigen values in descending order
    sortedIndex = np.argsort(eigenValues)[::-1]
    sortedValues = eigenValues[sortedIndex]
    sortedVectors = eigenVectors[:,sortedIndex]
     
    # Select first k eigen vectors
    selectedVectors = sortedVectors[:,:k]
     
    # reduce the dataset features
    X_reduced = np.dot(selectedVectors.transpose() , X.transpose() ).transpose()
     
    return X_reduced,eigenVectors
X_red,vectors = PCA(X_scaled, 5)
print(f"d. Reduced Datset: \n{X_red}")

# e. Build the Logistic regression model with the transformed dataset.
column1 = X_scaled.dot(vectors.T[0])
column2 = X_scaled.dot(vectors.T[1])
column3 = X_scaled.dot(vectors.T[2])
column4 = X_scaled.dot(vectors.T[3])
column5 = X_scaled.dot(vectors.T[5])
column6 = X_scaled.dot(vectors.T[6])
column7 = X_scaled.dot(vectors.T[7])
column8 = X_scaled.dot(vectors.T[8])
column9 = X_scaled.dot(vectors.T[9])
column10 = X_scaled.dot(vectors.T[10])
column11 = X_scaled.dot(vectors.T[11])
column12 = X_scaled.dot(vectors.T[12])
column13 = X_scaled.dot(vectors.T[13])
column14 = X_scaled.dot(vectors.T[14])

ds_pca = pd.DataFrame(column1, columns=['column1'])
ds_pca['column2'] = column2
ds_pca['column3'] = column3
ds_pca['column4'] = column4
ds_pca['column5'] = column5
ds_pca['column6'] = column6
ds_pca['column7'] = column7
ds_pca['column8'] = column8
ds_pca['column9'] = column9
ds_pca['column10'] = column10
ds_pca['column11'] = column11
ds_pca['column12'] = column12
ds_pca['column13'] = column13
ds_pca['column14'] = column14
ds_pca['Y'] = Y
print(ds_pca.head())

# Split the dataset into train-test
ds_pca = ds_pca.dropna()
X, Y = ds_pca.drop(columns = ['Y']), ds_pca['Y']
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2)

model = LogisticRegression()

# Fit the model on the training dataset
model.fit(X_train, Y_train)
accuracy = score = model.score(X_test, Y_test)
print(f'e. Accuracy of model is : {score*100} \n')