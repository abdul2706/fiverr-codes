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

def create_folds(folds, X, y):
    m, n = X.shape
    fold_size = int(m / folds)
    X_folds, y_folds = np.zeros((folds, fold_size, n)), np.zeros((folds, fold_size, 1))
    # print(X_folds.shape, y_folds.shape)
    for i in range(0, folds):
        fold_start = fold_size * i
        fold_end = fold_size * (i + 1)
        # temp_X = X[fold_start:fold_end]
        # temp_y = y[fold_start:fold_end]
        # print('[temp_X]', temp_X.shape, temp_y.shape)
        X_folds[i, :, :] = X[fold_start:fold_end]
        y_folds[i, :, :] = y[fold_start:fold_end].reshape([-1, 1])
    return X_folds, y_folds

# 1. Reads in the data, ignoring the first row (header) and first column (index).
dataset = pd.read_csv('x06Simple.csv', index_col=0)
X = dataset[['Temp of Water', 'Length of Fish']].values
y = dataset['Age'].values
m, n = X.shape
rmse = np.zeros(20)
folds_to_report = [4, 11, 22, m]
# folds_to_report = np.arange(4, m, 1)
# print(folds_to_report)

for S in folds_to_report:
    # 2. 20 times does the following
    for k in range(20):
        # (a) Shuffles the rows of the data
        indices_original = np.arange(len(X))
        indices_shuffled = np.random.permutation(indices_original)
        X = X[indices_shuffled]
        y = y[indices_shuffled]

        # (b) Creates S folds
        X_folds, y_folds = create_folds(S, X, y)
        
        # (c) for i = 1 to S
        for i in range(S):
            # i. Select fold i as your validation data and the remaining (S - 1) folds as your training data.
            X_train, y_train = np.concatenate(X_folds[np.arange(S) != i]), np.concatenate(y_folds[np.arange(S) != i])
            X_val, y_val = X_folds[i], y_folds[i]
        
            # ii. Train a linear regression model using the direct solution.
            model = linear_regression_direct(X_train, y_train)
            
            # iii. Compute the squared error for each sample in the current validation fold
            prediction_val = linear_regression_simple_predict(X_val, model)
            squared_error = np.square(y_val - prediction_val)
        
        # (d) You should now have N squared errors. Compute the RMSE for these.
        # print('[squared_error]', squared_error.shape)
        rmse[k] = np.sqrt(np.mean(squared_error))

    rmse_mean = rmse.mean()
    rmse_std = rmse.std()
    print(f'S = {S}, RMSE Mean = {rmse_mean}, RMSE_std = {rmse_std}')

    # break
