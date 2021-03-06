import numpy as np
import pandas as p
import math as m

def featureScaling(arr):
    max_value = max(arr)
    min_value = min(arr)
    denominator = max_value-min_value
    
    if denominator == 0:
        return [0.5 for i in range(len(arr))]
    
    # denominator needs to be (denominator*1.0) to force floating point division on Python 2, but not Python 3
    return [(arr[i]-min_value)/denominator for i in range(len(arr))]

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print(featureScaling(data))