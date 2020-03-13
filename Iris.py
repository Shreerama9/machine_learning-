#MULTIPLE REGRESSION 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error

iris = datasets.load_iris()

#print(iris.keys())
#dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
#print(iris.feature_names)

iris_x = iris.data


iris_x_train = iris.data[-100:]
iris_x_test = iris.data[:-50]

iris_y_train = iris.target[-100:]
iris_y_test = iris.target[:-50]

model = linear_model.LinearRegression()

model.fit(iris_x_train,iris_y_train)

iris_y_predict = model.predict(iris_x_test)


print("Mean Squared Error is ", mean_squared_error(iris_y_test,iris_y_predict))

#Wieghts
print("Weight : ",model.coef_)
#intercept
print("Interept : ",model.intercept_)


#Mean Squared Error is  0.3031383260611286
#Weight :  [-0.1960596  -0.30755035  0.38426438  0.68284465]
#Interept :  0.5813611362221778
