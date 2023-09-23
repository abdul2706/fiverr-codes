import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report

import tensorflow as tf

SUPPORTED_CLASSIFIERS = {'svc': SVC, 'knn': KNeighborsClassifier}


def train_classifier(X_train, y_train, classifier_name, **kwargs):
    # Instantiate the KNN classifier
    if classifier_name not in list(SUPPORTED_CLASSIFIERS.keys()):
        raise Exception(f'Invalid argument classifier={classifier_name}, supported classifiers are: {"".join(list(SUPPORTED_CLASSIFIERS.keys()))}')

    classifier_class = SUPPORTED_CLASSIFIERS[classifier_name]
    classifier = classifier_class(**kwargs)

    # Fit the classifier to the training data
    with tf.device('/gpu:0'):
        classifier.fit(X_train, y_train)

    return classifier


def test_classifier(classifier, X_train, y_train, X_test, y_test):
    # Calculate Accuracy
    training_accuracy = classifier.score(X_train, y_train)
    testing_accuracy = classifier.score(X_test, y_test)
    print('Training Accuracy:', training_accuracy)
    print('Test Accuracy:', testing_accuracy)

    # Get predictions on test dataset and show classification reoprt
    y_pred = classifier.predict(X_test)
    target_names = ['No Tumor', 'Tumor']
    print(f'******* Classification Report of {classifier.__class__.__name__} *******')
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    # plot normalized confusion matrix
    plot_confusion_matrix(classifier, X_test, y_test, normalize="true", cmap=plt.cm.Blues, display_labels=['Not tumor', 'Tumor'])
    plt.show()

    return training_accuracy, testing_accuracy


def train_test_KNN(X_train, y_train, X_test, y_test):
    # reshape dataset
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))
    print('X_train:', X_train.shape)
    # reshape dataset
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))
    print('X_test:', X_test.shape)

    classifier = train_classifier(X_train, y_train, 'knn', n_neighbors=5)
    test_classifier(classifier, X_train, y_train, X_test, y_test)


def train_test_SVC(X_train, y_train, X_test, y_test):
    # reshape dataset
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))
    print('X_train:', X_train.shape)
    # reshape dataset
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))
    print('X_test:', X_test.shape)

    classifier = train_classifier(X_train, y_train, 'svc')
    test_classifier(classifier, X_train, y_train, X_test, y_test)


def KNN_neighbor_search(X_train, y_train, X_test, y_test, max_neighbors=9):
    # reshape dataset
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))
    print('X_train_reshaped:', X_train_reshaped.shape)
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))
    print('X_test_reshaped:', X_test_reshaped.shape)

    # Test KNN classifier for different neighbor values.
    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, max_neighbors)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in tqdm(enumerate(neighbors), total=len(neighbors), position=0, leave=True):
        # Setup a k-NN Classifier with k neighbors: knn_classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)

        # Fit the classifier to the training data
        with tf.device('/gpu:0'):
            knn_classifier.fit(X_train_reshaped, y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn_classifier.score(X_train_reshaped, y_train)

        # Compute accuracy on the testing set
        test_accuracy[i] = knn_classifier.score(X_test_reshaped, y_test)

    # Generate plot
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
