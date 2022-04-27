# PCA from Scratch
#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# a. importing Dataset: IRIS
ds = pd.read_csv("Iris.csv")
print(f"a. Dataset: \n {ds.head()}\n")

# b. Scale the Dataset 
column_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
for i in (column_names):
    ds[i] = (ds[i] - ds[i].mean())/ds[i].std()
X = ds.iloc[:, 1:5]
Y = ds.iloc[:, 5]
print(f"b. Scaled Dataset:\n {Y}\n")

# c. Calculate the covariance matrix for the features in the dataset.
X = X.values
X = X - X.mean()
cov = np.cov(X, rowvar = False)
print(f"c. Covariance matrix: \n{cov}\n")

# d. Calculate the eigenvalues and eigenvectors for the covariance matrix.
eigen_values , eigen_vectors = np.linalg.eigh(cov)
print(f"d. Eigen values: {eigen_values}")
print(f"    Eigen vectors: {eigen_vectors}\n")

# e. Sort eigenvalues and their corresponding eigenvectors.
idx = eigen_values.argsort()[::-1]   
sorted_eigen_values = eigen_values[idx]
sorted_eigen_vectors = eigen_vectors[:,idx]
print(f"e. Sorted Eigen values: {sorted_eigen_values}")
print(f"    Sorted Eigen vectors: {sorted_eigen_vectors}\n")

# f. Plot the principal components and percentage of explained variances.
# To find the percentage of variance explained, divide each eigen value with the sum of all eigen values
explained_variances = []
for i in range(len(sorted_eigen_values)):
    explained_variances.append(sorted_eigen_values[i] / np.sum(sorted_eigen_values))
print(f"f. Percentage of explained variances: {explained_variances}")

# plottong PCA
eigen_vec_subset = sorted_eigen_vectors[:,0:2]
X_red = np.dot(eigen_vec_subset.transpose(), X.transpose()).transpose()
analyzed_data = pd.DataFrame(X_red, columns=['PC1', 'PC2'])
analyzed_data = pd.concat([analyzed_data, pd.DataFrame(Y)], axis=1)
plt.figure(figsize=(6, 6))
sb.scatterplot(data=analyzed_data, x='PC1', y='PC2', hue=Y, s=30)

# g. Choose first k eigen vectors
k = 3
k_eigen = sorted_eigen_values[:k], sorted_eigen_vectors[:,:k]
print(f"g. Choose first k eigen vectors: \n{k_eigen}\n")

# h. Transform the original matrix.
print(f"h. Transform the original matrix: \n{X_red}")