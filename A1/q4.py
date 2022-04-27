#defining 2 variables and taking user defined input and then typecasting to integer
x = int(input("Enter 1st Integer: "))
y = int(input("Enter 2nd Integer: "))

#using 3rd variable to store values temporarily and swapping
z = x 
x = y
y = z

print('After Swapping')
print('1st Integer: ',x)
print('2nd Integer: ',y)