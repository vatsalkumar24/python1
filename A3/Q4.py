# Q4 a) Binary search using recursion
def recursiveBS(num, l, r, x):
    if (l<=r):
        #mid index calculation and trying to find its exact location, if found return it else return -1
        mid = l + (r-l)//2
        if(num[mid] == x): return mid
        #if element is in left part of array from current mid index
        elif(num[mid] > x) : return recursiveBS(num, l, mid - 1, x)
        #if element is present in right part of array from current index
        else: return recursiveBS(num, mid + 1, r, x)
    else: return -1
# Q4 b) Binary search using iteration
def iterativeBS(num, l, r, x):
    
    while(l<=r): 
        #mid index calculation and trying to find its exact location, if found return it else return -1
        mid = l + (r-l)//2
        if(num[mid] == x): return mid
        #if element is in left part of array from current mid index
        elif(num[mid] > x): r = mid-1
        #if element is present in right part of array from current index
        else: l = mid+1
    return -1

#two user defined arrays for testign both methods
a = []
b = []
size1 = int(input("Enter size of 1st array: "))
size2 = int(input("Enter size of 2nd array: "))

#taking input usng map
a = list(map(int,input("Enter 1st Array: ").strip().split()))[:size1]
b = list(map(int,input("Enter 2nd Array: ").strip().split()))[:size2]

#calling search functions to find user entered element, if not found print apporiate message
x = int(input("Q4 (a)Enter a number to search in Array 1(using Recursion): "))
pos = recursiveBS(a,0,size1 -1,x)
if(pos == -1): print("Element not found!")
else: print(f"Element found at index: {pos}")

x = int(input("Q4 (b)Enter a number to search in Array 2(using Iteration): "))
pos = iterativeBS(b,0,size2 -1,x)
if(pos == -1): print("Element not found!")
else: print(f"Element found at index: {pos}")




