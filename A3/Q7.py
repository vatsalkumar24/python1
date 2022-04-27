#Q7 Salary Dataset Operations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# a. Reading the dataset and displaying using panadas read_csv
ds = pd.read_csv("./Salary_Data.csv")
print(f"a. Salary Dataset: \n{ds}\n")

# b. Display the information related to the dataset such as the number of rows and columns using shape()
print(F"b. Information related to the dataset [ROWS, COLUMNS]: {ds.shape}\n")

# c. displaying first 5 rows using head()
print(f"c.First 5 rows: \n{ds.head()}\n")

# d. Displaying the summary statistics for each numeric column
print("d. Summary statistics: ")
print(ds.describe(),"\n")

# e. Displaying a random sample 
print("e. Display a random sample : ")
print(ds.sample(5),"\n")