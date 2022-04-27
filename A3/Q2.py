#Q2 bubble sort 



#defining global list and its size(default is 0)
num = []
size = 0
#global pass counter
count = 1
#function to bubble sort the list
def bubblesort():   
    #taking array elements from user
    num = list(map(int,input("Enter the numbers : ").strip().split()))[:size]

    #looping through all elements of list
    for x in range(size):
        #looping though first (size - 1 - x) elements and swaping them if needed, last x elements are sorted
        for y in range(size - 1 - x):
            if(num[y] > num[y+1]):
                num[y] , num[y+1] = num[y+1] , num[y]
    
    print(f"Array after bubble sorting: {num}\n")

while True:
    #pass number
    print(f"Pass No. : {count} ")
    count += 1

    #taking array size from user, if -1 is entered exits the program
    size = int(input("Enter the size of array: "))
    if(size != -1):
        bubblesort()
    else: break
    
