from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

#print(iris.keys())
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(iris.DESCR)
# print(iris.feature_names[0],iris.target[0])
#print(iris.data)


features = iris.data
labels = iris.target

clf = KNeighborsClassifier()
clf.fit(features, labels)

a = float(input())  # a is sepal length in cm
b = float(input())  # b is sepal width in cm
c = float(input())  # c is petal length in cm
d = float(input())  # d is petal width in cm

predc = clf.predict([[a, b, c, d]])
#print(predc)

if predc == [0]:
    print("Iris-Setosa")
elif predc == [1]:
    print("Iris-Versicolour")
else:
    print("Iris-Virginica")
