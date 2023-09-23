from cv2 import threshold
import numpy as np
import pandas as pd

np.random.seed(0)

def RMSE(y, prediction):
    return np.sqrt(np.mean(np.square(y - prediction)))

def MAPE(y, prediction):
    return np.mean(np.abs((y - prediction) / y))

# Write a script that:
# 1. Reads in the data, ignoring the first row (header) and first column (index).
dataset = pd.read_csv('x06Simple.csv', index_col=0)
X = dataset[['Temp of Water', 'Length of Fish']].values.astype(np.float32)
y = dataset['Age'].values.astype(np.float32)
m, n = X.shape
X = np.append(X, np.ones((m, 1)), axis=1)
y = y.reshape(m, 1)

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

# 4. Zscores the features data based on the training data
means = np.zeros(n)
stds = np.zeros(n)
for i in range(n):
    means[i] = X_train[:, i].mean()
    stds[i] = X_train[:, i].std()
    X_train[:, i] = (X_train[:, i] - means[i]) / stds[i]
    X_val[:, i] = (X_val[:, i] - means[i]) / stds[i]

# 5. While the termination criteria (mentioned next in the implementation details) hasn't been met
#     (a) Compute the RMSE and MAPE for both the training and validation data using the current model.
#     (b) Update your parameters.
epoch = 0
learning_rate = 1e-4
# equivalent to 2.3283064e-10
absolute_percent_change_rmse_threshold = 2**-32
prev_rmse_train = 1e-10
absolute_percent_change_rmse = 1.0
# initialize the parameters using random values in the range -10^-4 to +10^-4
theta = np.random.uniform(low=-1e-4, high=1e-4, size=(1, X.shape[1]))

while epoch < 1e6 and absolute_percent_change_rmse > absolute_percent_change_rmse_threshold:
    # calculate model prediction
    y_hat = np.dot(X_train, theta.T)
    # calculate error
    error = y_train - y_hat
    # apply Gradient Descent to update parameters
    delta = -2 / n * np.dot(X_train.T, error)
    theta = theta - learning_rate * np.sum(delta)

    # Compute the RMSE and MAPE for both the training and validation data using the current model.
    prediction_train = np.dot(X_train, theta.T)
    prediction_val = np.dot(X_val, theta.T)
    rmse_train = RMSE(y_train, prediction_train)
    mape_train = MAPE(y_train, prediction_train)
    rmse_val = RMSE(y_val, prediction_val)
    mape_val = MAPE(y_val, prediction_val)
    absolute_percent_change_rmse = np.abs((rmse_train - prev_rmse_train) / prev_rmse_train)
    prev_rmse_train = rmse_train

    if epoch % 100 == 0 and epoch > 0:
        print(f'epoch={epoch:>6} | rmse_train={rmse_train:.4f} | mape_train={mape_train:.4f} | rmse_val={rmse_val:.4f} | mape_train={mape_train:.4f}')

    epoch += 1

print(f'epoch={epoch:>6} | rmse_train={rmse_train:.4f} | mape_train={mape_train:.4f} | rmse_val={rmse_val:.4f} | mape_train={mape_train:.4f}')
print(f'absolute_percent_change_rmse={absolute_percent_change_rmse}')
print('model: y = ', end='')
for i, w in enumerate(theta.flatten()):
    if i < len(theta.flatten()) - 1:
        print(f'{w} * x{i + 1}', end='')
        print(' + ', end='')
    else:
        print(f'{w}')
