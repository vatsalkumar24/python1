#Q1 LIST OPERATIONS

#list intitialised with different data
mylist = ["vatsal",24, 3.14,'hi',0,9.8]

#a) Count the length of the list
size = len(mylist)
print("a) Length of list: ",size,"\n")

#b) Access the last element in the list using negative indexing.
lastelement = mylist[-1]
print("b) Last element of list: ",lastelement,"\n")

#c) Add one item to a list using the append()method.
mylist.append("sojitra")
print("c) Added 1 item using append(): ", mylist,"\n")

#d) Add several items using the extend()method.
mylist.extend([1000, 'hello', 23.5])
print("d) Added several items usng extend(): ", mylist,"\n")

#e) Add a list as an item to the existing list (nested list).
newlist = ["nitw", 2023, 8.12]
mylist.insert(1,newlist)
print(f"e) Added {newlist} to existing list(nested list): ",mylist,"\n")

#f) accessing 3 elements from the list using their indexes
index1 = 1
index2 = 6
index3 = 0
print('f) Using index operator to acces items as followed')
print(f'  Item at index {index1}: {mylist[index1]}')
print(f'  Item at index {index2}: {mylist[index2]}')
print(f'  Item at index {index3}: {mylist[index3]}\n')

#g) inserting element at indexr 4
pos = 4
mylist.insert(pos, 9.99999999)
print(f"g) Added new element at index {pos} using insert(): ", mylist,"\n")

#h) replacing element at index 3
mylist[3] = "PI"
print("h) List after replacing at index 7: ",mylist,"\n")

#i) adding duplicate item at end of list
mylist.append(mylist[7])
print("i) Added duplicate element: ", mylist,"\n")

#j) removing elements using pop()
pos = 5
mylist.pop(pos)
print(f"j) Removed item at index {pos} using pop(): {mylist}\n")

#k) creating a new list of numbers and sorting it using sort()
num = [ 5,1,6,2,7,8,9]
num.sort()
print("k) Sorted a list in ascending order: ",num)

#l) reversing list using reverse()
print("l) Reversing list using reverse()")
print("  List before reversing: ",mylist)
mylist.reverse()
print("  List after reversing: ", mylist,"\n")