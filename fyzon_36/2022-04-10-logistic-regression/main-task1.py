import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

def confusion_matrix(y_true, y_pred):
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    return np.array([[tp, fp], [fn, tn]])

def compute_metrics(confusion_matrix):
    tp, fp, fn, tn = confusion_matrix.ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score, 'accuracy': accuracy}

def logistic_regression(X, theta):
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    # calculate model prediction
    z = np.dot(X, theta)
    h = sigmoid(z)
    return h

def log_loss(y, h):
    m = y.shape[0]
    h = np.where(h > 0.00001, h, 0.00001)
    h = np.where(h < 0.99999, h, 0.99999)
    vector = y * np.log(h) + (1 - y) * np.log(1 - h)
    loss = -np.sum(vector) / m
    return loss

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

# 1. Reads in the data.
# dataset_path = 'Datasets/iris.data'
dataset_path = 'Datasets/spambase.data'
df_dataset = pd.read_csv(dataset_path, sep=',', header=None)
# print(df_dataset)
# print(df_dataset.iloc[:, -1].value_counts())
X = df_dataset.iloc[:, :-1].values
y = df_dataset.iloc[:, -1].values
m, n = X.shape

# 2. Shuffles the observations.
indices_original = np.arange(len(X))
indices_shuffled = np.random.permutation(indices_original)
X = X[indices_shuffled]
y = y[indices_shuffled]

# 3. Selects the first 2/3 (round up) of the data for training and the remaining for validation.
train_val_ratio = 2/3
total_rows = len(X)
X_train, y_train = X[:int(total_rows * train_val_ratio)], y[:int(total_rows * train_val_ratio)]
X_val, y_val = X[int(total_rows * train_val_ratio):], y[int(total_rows * train_val_ratio):]
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

# 4. Zscores the features based on the training data.
means = np.zeros(n)
stds = np.zeros(n)
for i in range(n):
    means[i] = X_train[:, i].mean()
    stds[i] = X_train[:, i].std()
    X_train[:, i] = (X_train[:, i] - means[i]) / stds[i]
    X_val[:, i] = (X_val[:, i] - means[i]) / stds[i]

# 5. Trains a logistic regression model using gradient descent, keeping track of the mean log loss of the training and validation sets as it trains.
epoch = 0
learning_rate = 0.1
# initialize the parameters using random values in the range -10^-4 to +10^-4
theta = np.random.uniform(low=-1e-4, high=1e-4, size=(X.shape[1], 1))
log_loss_train = []
log_loss_val = []

while epoch < 2500:
    # calculate model prediction
    h_train = logistic_regression(X_train, theta)
    
    # apply Gradient Descent to update parameters
    gradient = np.dot(X_train.T, h_train - y_train) / m
    theta = theta - learning_rate * gradient

    # Keep track of the mean log loss of the training and validation sets as it trains.
    h = logistic_regression(X_train, theta)
    log_loss_train.append(log_loss(y_train, h))
    h = logistic_regression(X_val, theta)
    log_loss_val.append(log_loss(y_val, h))

    if epoch % 100 == 0 and epoch > 0:
        mean_train_loss = np.mean(log_loss_train)
        mean_val_loss = np.mean(log_loss_val)
        print(f'epoch={epoch:>6} | mean_train_loss={mean_train_loss:.4f} | mean_val_loss={mean_val_loss:.4f}')

    epoch += 1

mean_train_loss = np.mean(log_loss_train)
mean_val_loss = np.mean(log_loss_val)
print(f'epoch={epoch:>6} | mean_train_loss={mean_train_loss:.4f} | mean_val_loss={mean_val_loss:.4f}')

# 6. Plots the training and validation mean log loss as a function of the epoch.
epochs = range(epoch)
fig1 = plt.figure()
fig1.suptitle('Train Loss and Val Loss')
plt.plot(epochs, log_loss_train, label='Train Loss')
plt.plot(epochs, log_loss_val, label='Val Loss')
plt.legend()

# 7. Computes the precision, recall, f-measure and accuracy of the learned model on the training and validation sets when using a threshold of 0.5.
h = logistic_regression(X_train, theta)
y_pred = h >= 0.5
cm_train = confusion_matrix(y_train, y_pred)
train_metrics = compute_metrics(cm_train)
print('Train set metrics:', train_metrics)

h = logistic_regression(X_val, theta)
y_pred = h >= 0.5
cm_val = confusion_matrix(y_val, y_pred)
val_metrics = compute_metrics(cm_val)
print('Val set metrics:', val_metrics)

# 8. Plots a precision-recall graph by varying the threshold from 0.0 to 1.0, inclusive, in increments of 0.1.
fig2 = plt.figure()
fig2.suptitle('Precision-Recall Curve of Val Data')
thresholds = np.arange(0, 1.01, 0.01)
h = logistic_regression(X_val, theta)
precisions, recalls = [], []
for threshold in thresholds:
    y_pred = h >= threshold
    cm_val = confusion_matrix(y_val, y_pred)
    val_metrics = compute_metrics(cm_val)
    precisions.append(val_metrics['precision'])
    recalls.append(val_metrics['recall'])
plt.plot(recalls, precisions)
fig1.savefig('train-val-loss-curve.jpg', dpi=300)
fig2.savefig('pr-curve.jpg', dpi=300)
plt.show()
