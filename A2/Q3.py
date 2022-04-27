#Q3 Set operations

#creating two sets
s1 = set((24,2,1,6,999,4))
s2 = set((4,458,24,945,2,100))

#printing sets before operations
print(f'Set S1: {s1}')
print(f'Set S2: {s2}\n')

#a) Performing union and intersection
print("a) Union and Intersection of sets")
#using union()
print(f'  Union of S1 and S2: {s1.union(s2)}')
#using intersection
print(f"  Intersection of S1 and S2: {s1.intersection(s2)}\n")

#b)add() and update() elements
print(f"b) Adding and updating elements: ")
s1.add(695)
print(f"  Element added to s1: {s1}")
s2.update({659 : 'six fifty nine'})
print(f"  Element updated to s2: {s2}\n")

#c) Perform S1-S2
print("c) S1-S2: ",s1.difference(s2),"\n")

#d) Symmetric Difference of S1 and S2
print(f"d) Symmetric Difference of S1 and S2: {s1.symmetric_difference(s2)}")