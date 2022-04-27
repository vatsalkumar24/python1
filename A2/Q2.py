#Q2 creating tuple with 5 elements and implementing count() and index()

#creating my own tuple
mytuple = (4,3,1,3,6,2)

#implementing count and returning count of 2 in above tuple
def mycount(num):
    return mytuple.count(num)

#implementing index and returning index of 6 in above tuple
def myindex(num):
    return mytuple.index(num)

#printing tuple
print("My tuple is: ",mytuple)

#executing mycount() and finding the count of 3
print("Count of 2: ",mycount(3))

#executing myindex and finding the index of 6
print("Index of 6: ",myindex(6))