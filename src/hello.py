import numpy as np
import pandas as p
from IPython.display import Filelink, FileLinks

msg = "Hello world"
print(msg)


x = np.array([[0, 1, 2, 3, 4, 5], [10, 20, 30, 40, 50, 60], [10, 20, 30, 40, 50, 60]])

print(x[-2:])


#Given a Pandas dataframe 'df' with columns 'gender' and 'age', how would you compute the average age for each gender?
#df.groupby('gender')['age'].mean()

#Which of the following commands would you use to visualize the distribution of 'height' values in a Pandas dataframe 'df'?
#df['height'].plot(kind='box')