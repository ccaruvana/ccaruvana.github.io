import math
import matplotlib.pyplot as plt
from sys import stdin
import numpy as np

def dot(xArr,yArr):
    return sum([x*y for x,y in zip(xArr,yArr)])

def p(wArr,xArr):
    temp = math.exp(dot(wArr[:-1:],xArr) + wArr[-1])
    return temp/(1+temp)

#   Prepare the input
inputArray = list(stdin.readlines())
inputArray = [line.strip().split() for line in inputArray]
inputArray = [[float(data) for data in line] for line in inputArray]
inputArray = np.array(inputArray)

#   We set up the weights
k = len(inputArray[0])
W = [0 for j in range(k)]

#   This portion is just the gradient descent with a scaling constant of 1
for index in range(100):
    pArr = [p(W,x[:-1]) for x in inputArray]
    for step in range(k-1):
        W[step] += sum([x*(y-p) for x,y,p in zip(inputArray[:,step],inputArray[:,-1],pArr)])
    W[k-1] += sum([y - p for y,p in zip(inputArray[:,-1],pArr)])

#   The folloing items can be used to separate the blues from the reds to aid in visual representation
blue = []
red = []
for line in inputArray:
    if line[-1] == 0:
        blue.append(line[:-1])
    else:
        red.append(line[:-1])
blueLen = len(blue)
redLen = len(red)

print("\nThe Blues\t\tThe Reds")
for index in range(max(blueLen,redLen)):
    if index < blueLen and index < redLen:
        print(str(blue[index])+"\t\t"+str(red[index]))
    elif index < blueLen:
        print(str(blue[index])+"\t\t---")
    elif index < redLen:
        print("---\t\t"+str(red[index]))

#   Display the separating plane
print("\nThe separating plane is given by\n")
print("0 = ",end="")
for index in range(len(W)-1):
    print("("+str(W[index])+") X"+str(index+1),end=" + ")
print("("+str(W[-1])+")"+"\n\n")


exit()
