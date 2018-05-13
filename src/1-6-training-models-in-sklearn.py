import pandas
import numpy

# Read the data
data = pandas.read_csv('1-6-data.csv')

# Split the data into X and y
X = numpy.array(data[['x1', 'x2']])
y = numpy.array(data['y'])

# import statements for the classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# TODO: Pick an algorithm from the list:
# - Logistic Regression
#classifier = LogisticRegression()

# - Decision Trees
classifier = DecisionTreeClassifier()

# - Support Vector Machines
#classifier = SVC()

# Define a classifier (bonus: Specify some parameters!)
# and use it to fit the data
# Click on `Test Run` to see how your algorithm fit the data!
classifier.fit(X, y)