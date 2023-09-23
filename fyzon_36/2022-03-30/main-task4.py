import numpy as np
import pandas as pd

np.random.seed(0)

def compute_weights(X_train, X_val_sample):
    k = 1
    m = X_train.shape[0] 
    weight_matrix = np.mat(np.eye(m))
    for i in range(m):
        xi = X_train[i]
        l1_distance = np.abs(xi - X_val_sample)
        weight_matrix[i, i] = np.exp( -(np.dot(l1_distance.T, l1_distance)) / k**2 )
    return weight_matrix

def locally_weighted_linear_regression(X, y, weights):
    m = X.shape[0]
    # Appending a cloumn of ones in X to add the bias term
    X = np.append(X, np.ones(m).reshape(m,1), axis=1)
    y = y.reshape((-1, 1))
    # Calculating parameter theta using the formula.
    theta = np.linalg.pinv(X.T * (weights * X)) * (X.T * (weights * y))
    return np.asarray(theta)

def locally_weighted_linear_regression_predict(X, theta):
    # Number of training examples
    # m = X.shape[0]
    # Appending a cloumn of ones in X to add the bias term.
    # X = np.append(X, np.ones((m, 1)), axis=1)
    # print(X.shape)
    X = np.append(X, np.ones(1), axis=0)
    # print(X.shape)
    # preds is y_hat which is the dot product of X and theta.
    preds = np.dot(X, theta)
    return preds

def RMSE(y, prediction):
    return np.sqrt(np.mean(np.square(y - prediction)))

def MAPE(y, prediction):
    return np.mean(np.abs((y - prediction) / y))

# def compute_weights(X_train, X_val_sample):
#     # print('[compute_weights]', X_train.shape, X_val_sample.shape)
#     k = 1
#     # calculate L1 distance
#     distances = np.sum(np.abs(X_train - X_val_sample), axis=1)
#     # print('[compute_weights][distances]', distances.shape)
#     # calculate weights using similarity function
#     weight_matrix = np.exp(-np.square(distances) / k**2)
#     # print('[compute_weights][weight_matrix]', weight_matrix.shape)
#     return weight_matrix

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
squared_error = []

# 4. Then for each validation sample
for X_val_sample, y_val_sample in zip(X_val, y_val):
    # (a) Compute the necessary distances of the validation sample to the training data in order to establish your weight matrix.
    weights = compute_weights(X_train, X_val_sample)

    # (b) Use the weight matrix to compute a local model via the direct method.
    model = locally_weighted_linear_regression(X_train, y_train, weights)
    # print('model: y = ', end='')
    # for i, w in enumerate(model):
    #     # print(w)
    #     if i < len(model) - 1:
    #         print(f'{w[0]:.4f} * x{i + 1}', end='')
    #         print(' + ', end='')
    #     else:
    #         print(f'{w[0]:.4f}')
    
    # (c) Evaluate the validation sample using the local model.
    prediction_val = locally_weighted_linear_regression_predict(X_val_sample, model)

    # (d) Compute the squared error of the validation sample.
    squared_error.append(y_val - prediction_val)

# 5. Computes the RMSE and MAPE over the validation data.
squared_error = np.array(squared_error)
rmse_val = np.sqrt(np.mean(squared_error))
print('Validation Set - RMSE:', rmse_val)
mape_val = np.mean(np.abs(squared_error / y_val))
print('Validation Set - MAPE:', mape_val)
