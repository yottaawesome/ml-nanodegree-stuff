# TODO: Add import statements
import pandas
import numpy
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pandas.read_csv('3-1-1-linear-regression-in-scikitlearn.csv') 
X = numpy.array(bmi_life_data[['BMI']])
y = numpy.array(bmi_life_data[['Life expectancy']])

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(X, y)

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)
