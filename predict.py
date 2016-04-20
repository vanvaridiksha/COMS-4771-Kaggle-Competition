import numpy as np
import pandas as pd
import re
from sklearn import tree

input_file = "data/data.csv"

field_types_file = "data/field_types.txt"
D = {}

with open(field_types_file, "r") as content:
    array = []
    for line in content:
        temp_array = line.split()
        category_array = temp_array[1:len(temp_array)]
        D[temp_array[0]] = category_array
        print temp_array[0]
        print D[temp_array[0]]
# comma delimited is the default

df = pd.read_csv(input_file, header = 0)

#print df.dtypes;


# remove the non-numeric columns
#df = df._get_numeric_data()

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
