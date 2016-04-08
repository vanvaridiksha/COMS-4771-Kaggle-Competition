import numpy as np
import pandas as pd
from sklearn import tree

input_file = "data/data.csv"

# comma delimited is the default
df = pd.read_csv(input_file, header = 0)

print df.dtypes;


# remove the non-numeric columns
df = df._get_numeric_data()

# put the numeric column names in a python list
numeric_headers = list(df.columns.values)

# create a numpy array with the numeric values for input into scikit-learn
numpy_array = df.as_matrix()

(rowSize,columnSize) = numpy_array.shape

#X = numpy_array [0::, 0:52]
#Y = numpy_array [0, 52:53]
X = numpy_array [0::, 0:columnSize-1]
Y = numpy_array [0::, columnSize-1:columnSize]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

print clf.predict(X)
