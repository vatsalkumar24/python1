#Q3 Sum of two Matrices

#no.of rows and columns
R = int(input("Enter the number of rows: "))
C = int(input("Enter the number of columns: "))

# Initializing 2 empty matrix and 1 empty result matrix
A = []
B = []
sum = []
print("Enter elements into matrix A:")
# taking single-single 1D lists and appending to Matrices
for i in range(R):          
    a = list(map(int,input().strip().split()))[:C]
    A.append(a)

print("Enter elements into matrix B:")
for i in range(R):          
    a = list(map(int,input().strip().split()))[:C]
    B.append(a)
#computing sum of 2 matrices
for i in range(R):		 
	a =[]
	for j in range(C):	 
		a.append(A[i][j] + B[i][j])
	sum.append(a)

# Result Matrix
print("Sum Matrix: ")
print(sum)
