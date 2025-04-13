import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
X = np.c_[np.ones((len(x_train), 1)), x_train]
theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_train)

# TODO: calculate error
mse = np.mean((y_test - (theta_best[0] + theta_best[1] * x_test)) ** 2)
print("Error mse:", mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
#  np.mean - srednia z populacji
#  np.std - odchylenie standardowe populacji
x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

x_train_standardized = (x_train - x_mean) / x_std
x_test_standardized = (x_test - x_mean) / x_std

y_train_standardized = (y_train - y_mean) / y_std
y_test_standardized = (y_test - y_mean) / y_std

X_standardized = np.c_[np.ones((len(x_train_standardized), 1)), x_train_standardized]

# TODO: calculate theta using Batch Gradient Descent

theta = [0.5, 0.5]
learning_rate = 0.01
iterations = 1000
m = len(x_train_standardized)

for iteration in range(iterations):
    gradients = 2/m * X_standardized.T.dot(X_standardized.dot(theta) - y_train_standardized)
    theta = theta - learning_rate * gradients


# TODO: calculate error
mse = np.mean((y_test_standardized - (theta[0] + theta[1] * x_test_standardized)) ** 2)
print("Error mse:", mse)

# plot the regression line for closed-form solution
x = np.linspace(min(x_test_standardized), max(x_test_standardized), 100)
y = theta[0] + theta[1] * x
plt.plot(x, y)
plt.scatter(x_test_standardized, y_test_standardized)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Regression Line')
plt.show()


