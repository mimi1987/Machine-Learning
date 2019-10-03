from sklearn import tree

# Training data
features = [[140, 1],
            [130, 1],
            [150, 0],
            [170, 0]]
labels = [1, 1, 0, 0]

# Train classifier (decision tree)
clf = tree.DecisionTreeClassifier()

# Training algorithm included in the classifier (fit)
clf = clf.fit(features, labels)

# Give a new imput to classifier. The input is a new example
print(clf.predict([[150, 0]]))
