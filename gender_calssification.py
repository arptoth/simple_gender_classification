from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Data for X_testing
# [height, weight, shoe_size]
X_test = [
    [188, 70, 38], [189, 94, 34], [180, 70, 42], [159, 61, 37], [165, 58, 39],
    [162, 54, 34], [171, 60, 40], [170, 70, 40], [143, 45, 37], [153, 48, 39]
]

Y_test = ['male', 'male', 'male', 'female', 'male',
          'female', 'male', 'male', 'female', 'female']


# Decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

prediction = clf.predict(X_test)

print("Decision tree: ", prediction)

# Support Vector Machine
clf = svm.SVC()
clf = clf.fit(X, Y)

prediction = clf.predict(X_test)

print("SVM: ", prediction)


# Random Forest
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)


prediction = clf.predict(X_test)

print("Random forest: ", prediction)

# print(clf.predict_proba(X_test))
