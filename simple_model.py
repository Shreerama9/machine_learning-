#LINEAR REGRESSION

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error

# a and b are random arrays created by me😅

a = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]])
b = np.array([[3],[2],[4],[8],[7],[1],[9],[5],[6]])

a_train = a
a_test = a

b_train = b
b_test = b

model = linear_model.LinearRegression()

model.fit(a_train,b_train)

b_predict = model.predict(a_test)


print("Mean Squared Error is ", mean_squared_error(b_test,b_predict))

#Wieghts
print("Weight : ",model.coef_)
#intercept
print("Interept : ",model.intercept_)

plt.scatter(a_test,b_test)
plt.plot(a_test,b_predict)

plt.show()
