#Q1 Prime numberser between 1 and 1000 using loop

#range
low = 1
high = 1000

print(f"Prime numbers between {low} & {high}: \n")

#using simple algorithm to loop through all the number 
#from 1 to 1000 and checking for each if any divisors present other tha 1 and itself
print(2,end = " ")
for x in range(low, high+1,2):
    if (x > 1):
        for y in range(2,x):
            if(x%y == 0):
                break
        else: print(x,end = " ")
print("\n")