import numpy as np
import math as m

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    ret = []
    i = 0
    totalExp = 0.
    while i < len(L):
        totalExp+=m.e**L[i]
        i+=1
    i = 0    
    while i < len(L):
        ret.append((m.e**L[i])/totalExp)
        i+=1
    return ret

def softmax2(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
    
    # Note: The function np.divide can also be used here, as follows:
    # def softmax(L):
    #     expL = np.exp(L)
    #     return np.divide (expL, expL.sum())