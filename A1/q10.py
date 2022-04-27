#importing math library
import math

#using function to find area of triangle
#using base and height
def area1(base, height):
    return 1/2* base * height

#using 3 sides of triangle
def area2(a,b,c):
    s = (a + b + c)/2
    return math.sqrt((s*(s-a)*(s-b)*(s-c)))

#taking input     
b = int(input('Enter base of triangle: '))
h = int(input('Enter height of triangle: '))

#calling area1()
print('Area of Triangle is ',area1(b,h))

#taking input  
a = int(input('Enter 1st side of triangle: '))
b = int(input('Enter 2nd side of triangle: '))
c = int(input('Enter 3rd side of triangle: '))

#calling area2()
print('Area of Triangle is ',area2(a,b,c))