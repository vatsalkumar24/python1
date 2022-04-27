#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# LDA transform function
def ldaTransform(no_comp, X_train, Y_train):
    m, n = X_train.shape
    uniq_classes = np.unique(Y_train)

    # calculating the scatter_t of the X_train
    sc_t = np.cov(X_train.T)*(m-1)

    # separating the rows according to the class
    # calculating the scatter_w of each class in X_train
    sc_w = 0
    for i in range(len(uniq_classes)):
        items = np.flatnonzero(Y_train==uniq_classes[i])
        sc_w += np.cov(X_train[items].T)*(len(items)-1)
    
    # Get the eigen values & vectors for scatter matrices
    sc_b = sc_t - sc_w
    eigVal, eigVect = np.linalg.eigh(np.linalg.pinv(sc_w).dot(sc_b))
    
    # Transforming the X_train (transforming all the features)
    X_train_transformed = X_train.dot(eigVect[:,::-1][:,:no_comp])

    return X_train_transformed


ds = pd.read_csv("wine.csv")
X = ds.iloc[:, 0:13]
Y = ds.iloc[:, 13]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# Normalise the X data
Scaler = StandardScaler()
Scaler.fit(X_train)
X_train = Scaler.transform(X_train)
Scaler.fit(X_test)
X_test = Scaler.transform(X_test)

# Transform the data using LDA
X_train_transformed = ldaTransform(9, X_train, Y_train)

print("X_train Dimensions before LDA transformation:", X_train.shape)
print("X_train Dimensions after LDA transformation:", X_train_transformed.shape)

# Now fit a logistic regression to the transformed wine dataset
model = LogisticRegression()
model.fit(X_train_transformed, Y_train)

# Make predictions using the model
X_test_transformed = ldaTransform(9, X_test, Y_test)
preds = model.predict(X_test_transformed)
print("prediction: ",preds)
print("Confusion Matrix: ",confusion_matrix(Y_test,preds))
