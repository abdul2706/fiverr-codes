import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

def sigmoid(z):
    """
    Logistic/Sigmoid Function:
        y = 1 / (1 + e^-z)
    """
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, theta):
    """
    This function calculates weighted sum of X and theta, then applies sigmoid function to calculate probability
    """
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

def train(model, X_train, y_train, X_val, y_val):
    """
    This function trains Logistic Regression model on two classes using Gradient Descent algorithm
    """
    epoch = 0
    learning_rate = 0.1
    log_loss_train = []
    log_loss_val = []
    theta = model['weights']
    c1, c2 = model['classes']
    print(f'Training Logistic Regression for classes {c1} and {c2}')
    
    while epoch < 1000:
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

    model['weights'] = theta
    mean_train_loss = np.mean(log_loss_train)
    mean_val_loss = np.mean(log_loss_val)
    print(f'epoch={epoch:>6} | mean_train_loss={mean_train_loss:.4f} | mean_val_loss={mean_val_loss:.4f}\n')

    return epoch, log_loss_train, log_loss_val

def select_samples(model, X_train, y_train, X_val, y_val):
    """
    Function to select samples of two classes only. It assigns label 0 to first class and label 1 to second class
    """
    # store class label to select samples for
    c1, c2 = model['classes']

    # find train set rows corresponding to classes c1 and c2
    rows_train = np.where(np.logical_or(y_train == c1, y_train == c2))[0]
    # select train set rows
    X_train_ = X_train[rows_train]
    y_train_ = y_train[rows_train].reshape(-1, 1)
    # assign label 0 to classe c1 and label 1 to class c2
    y_train_[np.where(y_train_ == c1)] = 0
    y_train_[np.where(y_train_ == c2)] = 1

    # find val set rows corresponding to classes c1 and c2
    rows_val = np.where(np.logical_or(y_val == c1, y_val == c2))[0]
    # select val set rows
    X_val_ = X_val[rows_val]
    y_val_ = y_val[rows_val].reshape(-1, 1)
    # assign label 0 to classe c1 and label 1 to class c2
    y_val_[np.where(y_val_ == c1)] = 0
    y_val_[np.where(y_val_ == c2)] = 1

    return X_train_, y_train_, X_val_, y_val_

def plot_loss_curves(log_loss_train, log_loss_val, epoch, filename):
    """
    Function to plot train and val loss curves generated during training process by `train` method
    """
    epochs = range(epoch)
    fig1 = plt.figure()
    fig1.suptitle('Train Loss and Val Loss')
    plt.plot(epochs, log_loss_train, label='Train Loss')
    plt.plot(epochs, log_loss_val, label='Val Loss')
    plt.savefig(filename, dpi=300)
    plt.legend()

def confusion_matrix(y_true, y_pred):
    """
    Calculates multiclass confusion matrix such that rows index correspond to predicted label 
    and columns index correspond to true label.
    ```
    e.g.,
                   True Labels
                    0   1   2
    Predicted   0   8   1   1
      Labels    1   1   9   3
                2   2   2   9
    ```
    """
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)))
    for i in classes:
        for j in classes:
            cm[i, j] = np.sum(np.logical_and(y_pred == i, y_true == j))
    return cm.astype(np.int16)

def print_confusion_matrix(cm):
    """
    Function to print 3x3 confusion matrix on console
    """
    classes = range(len(cm))
    print('\t\t\t\tTrue Class')
    print('\t\t\t', end='')
    for i in classes:
        print(f'{i:^8}', end='')
    print()
    for i in classes:
        if i != 1:
            print('\t\t', end='')
        if i == 1:
            print('Predicted Class ', end='')
        print(f'{i:^8}', end='')
        for j in classes:
            print(f'{cm[i, j]:^8}', end='')
        print()

# 1. Reads in the data.
class_to_label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
dataset_path = 'Datasets/iris.data'
df_dataset = pd.read_csv(dataset_path, sep=',', header=None)
df_dataset.iloc[:, -1] = df_dataset.iloc[:, -1].apply(lambda x: class_to_label[x])
X = df_dataset.iloc[:, :-1].values
y = df_dataset.iloc[:, -1].values
y = y.reshape(-1, 1)
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

# 4. Zscores the features based on the training data.
means = np.zeros(n)
stds = np.zeros(n)
for i in range(n):
    means[i] = X_train[:, i].mean()
    stds[i] = X_train[:, i].std()
    X_train[:, i] = (X_train[:, i] - means[i]) / stds[i]
    X_val[:, i] = (X_val[:, i] - means[i]) / stds[i]

# 5. Trains three models for one-vs-one multi-class classification, each of which:
# initialize the parameters using random values in the range -10^-4 to +10^-4
model_01 = {'weights': np.random.uniform(low=-1e-4, high=1e-4, size=(X.shape[1], 1)), 'classes': [0, 1]}
model_12 = {'weights': np.random.uniform(low=-1e-4, high=1e-4, size=(X.shape[1], 1)), 'classes': [1, 2]}
model_20 = {'weights': np.random.uniform(low=-1e-4, high=1e-4, size=(X.shape[1], 1)), 'classes': [2, 0]}

# (a) Selects the samples pertaining to the two classes your comparing.
X_train_01, y_train_01, X_val_01, y_val_01 = select_samples(model_01, X_train, y_train, X_val, y_val)
X_train_12, y_train_12, X_val_12, y_val_12 = select_samples(model_12, X_train, y_train, X_val, y_val)
X_train_20, y_train_20, X_val_20, y_val_20 = select_samples(model_20, X_train, y_train, X_val, y_val)

# (b) Trains a logistic regression model using gradient descent.
epoch_01, log_loss_train_01, log_loss_val_01 = train(model_01, X_train_01, y_train_01, X_val_01, y_val_01)
epoch_12, log_loss_train_12, log_loss_val_12 = train(model_12, X_train_12, y_train_12, X_val_12, y_val_12)
epoch_20, log_loss_train_20, log_loss_val_20 = train(model_20, X_train_20, y_train_20, X_val_20, y_val_20)

# plot_loss_curves(log_loss_train_01, log_loss_val_01, epoch_01, 'model_01')
# plot_loss_curves(log_loss_train_12, log_loss_val_12, epoch_12, 'model_12')
# plot_loss_curves(log_loss_train_20, log_loss_val_20, epoch_20, 'model_20')

# 6. Applies the models to each validation sample to determine the most likely class.
h_01 = logistic_regression(X_val, model_01['weights']).ravel()
h_12 = logistic_regression(X_val, model_12['weights']).ravel()
h_20 = logistic_regression(X_val, model_20['weights']).ravel()

y_val_pred = np.zeros_like(y_val)
for idx, (h1, h2, h0) in enumerate(zip(h_01, h_12, h_20)):
    # h1 -> likelihood of 1 between 0 and 1
    # h2 -> likelihood of 2 between 1 and 2
    # h0 -> likelihood of 0 between 2 and 0
    likelihood_0 = (h0 + (1 - h1)) / 2
    likelihood_1 = (h1 + (1 - h2)) / 2
    likelihood_2 = (h2 + (1 - h0)) / 2
    y_true = y_val[idx, 0]
    y_pred = np.argmax([likelihood_0, likelihood_1, likelihood_2])
    y_val_pred[idx, 0] = y_pred
    # print(f'idx={idx}, likelihood_0={likelihood_0:.4f}, likelihood_1={likelihood_1:.4f}, likelihood_2={likelihood_2:.4f}, y_pred={y_pred}, y_true={y_true}')

# 7. Computes the validation accuracy.
accuracy = np.sum(y_val_pred == y_val) / len(y_val)
print('Validation Accuracy =', accuracy)

# 8. Creates a confusion matrix for the validation data.
cm_val = confusion_matrix(y_val, y_val_pred)
print('Confusion Matrix for the Validation Data')
print_confusion_matrix(cm_val)
