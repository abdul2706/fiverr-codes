"""# Linear Regression (Baseline)"""

from dataset import train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

print('[train_dataset]', train_dataset.X.shape, train_dataset.y.shape)
print('[val_dataset]', val_dataset.X.shape, val_dataset.y.shape)
print('[test_dataset]', test_dataset.X.shape, test_dataset.y.shape)

def predict(model, X_test, y_test, tag):
    # make predictions using the testing set
    y_pred = model.predict(X_test)
    # calculate the mean squared error
    print(f'[{tag}] Mean squared error: {mean_squared_error(y_test, y_pred):.8f}')
    # calculate the coefficient of determination: 1 is perfect prediction
    print(f'[{tag}] Coefficient of determination: {r2_score(y_test, y_pred):.8f}')

def train_predict(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # train model
    model.fit(X_train, y_train)
    predict(model, X_train, y_train, 'TRAIN')
    predict(model, X_val, y_val, 'VAL')
    predict(model, X_test, y_test, 'TEST')

train_predict(LinearRegression(), train_dataset.X, train_dataset.y, val_dataset.X, val_dataset.y, test_dataset.X, test_dataset.y)

train_predict(Ridge(), train_dataset.X, train_dataset.y, val_dataset.X, val_dataset.y, test_dataset.X, test_dataset.y)

train_predict(Lasso(), train_dataset.X, train_dataset.y, val_dataset.X, val_dataset.y, test_dataset.X, test_dataset.y)

train_predict(ElasticNet(), train_dataset.X, train_dataset.y, val_dataset.X, val_dataset.y, test_dataset.X, test_dataset.y)
