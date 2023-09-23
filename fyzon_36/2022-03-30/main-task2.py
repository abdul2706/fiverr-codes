import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

np.random.seed(0)

def linear_regression_direct(X, y):
    # Number of training examples
    m = X.shape[0]
    # Appending a cloumn of ones in X to add the bias term
    X = np.append(X, np.ones((m, 1)), axis=1)
    # reshaping y to (m,1)
    y = y.reshape(m,1)
    # The Normal Equation
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    return theta

def linear_regression_simple_predict(X, theta):
    # Number of training examples
    m = X.shape[0]
    # Appending a cloumn of ones in X to add the bias term.
    X = np.append(X, np.ones((m, 1)), axis=1)
    # preds is y_hat which is the dot product of X and theta.
    preds = np.dot(X, theta)
    return preds

def RMSE(y, prediction):
    return np.sqrt(np.mean(np.square(y - prediction)))

def MAPE(y, prediction):
    return np.mean(np.abs((y - prediction) / y))

# Write a script that:
# 1. Reads in the data, ignoring the first row (header) and first column (index).
dataset = pd.read_csv('x06Simple.csv', index_col=0)
X = dataset[['Temp of Water', 'Length of Fish']].values
y = dataset['Age'].values

# 2. Shuffles the rows of the data
indices_original = np.arange(len(X))
indices_shuffled = np.random.permutation(indices_original)
X = X[indices_shuffled]
y = y[indices_shuffled]

# 3. Selects the first 2/3 (round up) of the data for training and the remaining for validation.
train_val_ratio = 2/3
total_rows = len(X)
X_train, y_train = X[:int(total_rows * train_val_ratio)], y[:int(total_rows * train_val_ratio)]
X_val, y_val = X[int(total_rows * train_val_ratio):], y[int(total_rows * train_val_ratio):]

# 4. Computes the linear regression model using the direct solution.
model = linear_regression_direct(X_train, y_train)
print('model: y = ', end='')
for i, w in enumerate(model):
    if i < len(model) - 1:
        print(f'{w[0]:.4f} * x{i + 1}', end='')
        print(' + ', end='')
    else:
        print(f'{w[0]:.4f}')

# 5. Applies the learned model to the validation samples.
prediction_val = linear_regression_simple_predict(X_val, model)
prediction_train = linear_regression_simple_predict(X_train, model)

# 6. Computes the root mean squared error (RMSE) and mean absolute percent error (MAPE) for the training ...
rmse_train = RMSE(y_train, prediction_train)
print('Train Set - RMSE:', rmse_train)
mape_train = MAPE(y_train, prediction_train)
print('Train Set - MAPE:', mape_train)
# 6. ... and validation sets.
rmse_val = RMSE(y_val, prediction_val)
print('Validation Set - RMSE:', rmse_val)
mape_val = MAPE(y_val, prediction_val)
print('Validation Set - MAPE:', mape_val)
