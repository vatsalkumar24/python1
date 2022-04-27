# taking user defined input
year=int(input("Enter any year: "))

# checking conditions for leap year using if else and logical operators
if((year % 400 == 0) or (year % 100 != 0) and (year % 4 == 0)):  
    print("Its a Leap Year!")
else:
    print("Its not a Leap Year")