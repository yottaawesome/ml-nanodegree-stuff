import numpy as np
import pandas as p
import math as m


def sigmoid(x):
    return 1./(1+m.e**(-x))

def output_formula(features, weights, bias):
    i = 0
    cum = bias
    while i < len(features):
        cum += weights[i]*features[i]
        i+=1
    return sigmoid(cum)

def error_formula(y, output):
    return -y*m.log(output)-(1-y)*m.log(1-output)

def update_weights(x, y, weights, bias, learnrate):
    y_hat = output_formula(x, weights, bias)
    newBias = bias + learnrate*(y-y_hat)
    i = 0
    newWeights = []
    while i < len(weights):
        newWeights.append(weights[i]+learnrate*(y-y_hat)*x)
        i += 1
    return newWeights, newBias

y = 1
print(-y)

#msg = "Hello world"
#print(msg)

#x = np.array([[0, 1, 2, 3, 4, 5], [10, 20, 30, 40, 50, 60], [10, 20, 30, 40, 50, 60]])

#print(x[-2:])


#Given a Pandas dataframe 'df' with columns 'gender' and 'age', how would you compute the average age for each gender?
#df.groupby('gender')['age'].mean()

#Which of the following commands would you use to visualize the distribution of 'height' values in a Pandas dataframe 'df'?
#df['height'].plot(kind='box')