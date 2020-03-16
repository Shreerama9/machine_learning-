from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris["DESCR"])
x = iris["data"][:, 2:3]   # Petal Lenght
y = (iris["target"] == 2).astype(np.int)    # it helps to get 0 and 1 rather than false or true
clf = LogisticRegression()
clf.fit(x, y)

'''example_1 = (clf.predict([[4.6]]))
print(example_1)
example_2 = (clf.predict([[7.9]]))
print(example_2)'''

# Using matplotlib for Visualization
X = np.linspace(0, 9, 1000).reshape(-1, 1)
Y = clf.predict_proba(X)

plt.plot(X, Y[:,1])
plt.show()

