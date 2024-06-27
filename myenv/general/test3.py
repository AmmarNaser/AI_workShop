import matplotlib.pyplot as plt 
import numpy as nppip

x= range(1,50)
y= [value*3 for value in x]

print("Values of X: ")
print(range(1,50))
print (y)



plt.plot(x,y)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('draw a line')
plt.show()



