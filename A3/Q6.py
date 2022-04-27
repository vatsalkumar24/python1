# Q6 Student Data set creation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#creating a dataset for 7 days a week and its attendence out of 190
days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
attendence = []
for i in range(7):
    attd = np.random.randint(0,191)
    attendence.append(attd)

#creating dataframe using pandas
ds = pd.DataFrame({'Day' : days, 'Attendance' : attendence})

# a. displaying created dataset
print(f"a. Dataset: \n{ds}\n")

# b. Display the sorted dataset with least number of attendees at first
print(f"b. Sorted Dataset: \n{ds.sort_values('Attendance')}\n")

# c. Show the day with maximum number of attendees
max_attd = ds.iloc[ds.Attendance.argmax()]
print(f"c. Day with maximum attendence: \n{max_attd}\n")

#d. Display the 1st two days of the week and the number of attendees
print(f"d. First 2 days with attnedence: \n {ds[0:2]}\n")

# e. Plot the dataset for each day in the week.
ds.plot(x = 'Day',y = 'Attendance',kind='line')
print(f"e. Plotting Datase(): \n")
plt.show()