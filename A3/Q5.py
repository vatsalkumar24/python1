import numpy as np
#Q5 a) using numpy to create 2 1D arrays and find indices of those where arr1[i] > arr2[i]

# creating two lists of size n = 7
n = 7
a = [1,6,4,2,9,8,3]
b = [45,63,55,1,2,98,589]

# transforming list to array using numpy
arr1 = np.array(a)
arr2 = np.array(b)
print("Q5 (a)Indicies where elements of array 1 are greater than equal to that of array 2:",end=" ")
for i in range(n):
    if(arr1[i] >= arr2[i]):
        print(f"{i} ",end=" ")
print("")

#Q5 b) Operations on 1D 
print("Q5 (b)",end=" ")
a = np.array([2,6,19,27,15,43,124,129])

# i. replaced all even numbers with 0
a[a%2 == 0] = 0
print(f"i. Replaced all Even number with 0: {a}")

# ii. Extracting prime number from array using an algorithm
print("Q5 (b)",end=" ")
primes = []
for i in range(len(a)):
    isPrime = 1
    entered = 0
    if(a[i]>=2):
        entered = 1
        for j in range(2,a[i]):
            if(a[i]%j==0):
                isPrime = 0
                break
    if(isPrime==1 and entered==1):
        # insert a[i] into the set primes
        primes.append(a[i])
print(f"ii. Extracted Primes: {primes}")

# iii. Converting 1D to 2D using numpy reshape()
print("Q5 (b)",end=" ")
a2 = np.reshape(a,(2,a.size//2))
print(f"iii. Converted 1D to 2D array: [{a2[0]} {a2[1]}]")

# iv. Display the array element indices such that array elements are sorted in ascending order [ without the changing the position of elements]
print("Q5 (b)",end=" ")
print("iv. Sorted indices: ", np.argsort(a)) 

# v. Convert a binary NumPy array (holding only 0s and 1s) to a Boolean NumPy array.
#new numpy array containing  0s and 1s
binaryarray = np.array([0,1,1,0,1,0,1,1])
#new numpy array converted from above array
booleanarray = np.array(binaryarray,dtype = bool)
print("Q5 (b)",end=" ")
print(f"v. Converted  binary array {binaryarray} to boolean array {booleanarray}")

# vi. splitting input array into 3 parts
n = 10
#taking user defined input into listr
arr = list(map(int,input("Q5 (b) Enter 10 elements into Array: ").strip().split()))[:n]
#creating array from above list
a = np.array(arr)
a1 = np.split(a,[2,4])
print("Q5 (b)",end=" ")
print(f"vi. Splited array into 3 parts using NUMPY.SPLIT: {a1}")
