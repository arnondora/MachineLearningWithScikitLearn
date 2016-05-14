from sklearn import tree
features = [[320.5, 1], [315, 1], [210, 1], [350.2, 1], [300.1, 1], [12.2, 2], [14.3, 2], [10.5, 2], [11.2, 2], [11.1, 2], [20, 3], [21.2, 3], [22.2, 3], [23, 3]]
labels = [4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6]
classifier = tree.DecisionTreeClassifier()
classifier.fit(features, labels)
print(classifier.predict([[309.69, 1]]))
