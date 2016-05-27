import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

testDataIndex = [0, 50, 100]

# training set
train_target = np.delete(iris.target, testDataIndex)
trainData = np.delete(iris.data, testDataIndex, axis=0)

# test set
test_target = iris.target[testDataIndex]
test_data = iris.data[testDataIndex]

clf = tree.DecisionTreeClassifier()
clf.fit(trainData, train_target)

print(iris.target_names[clf.predict(test_data)])

import pydot
from sklearn.externals.six import StringIO 
from IPython.display import Image  
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  
