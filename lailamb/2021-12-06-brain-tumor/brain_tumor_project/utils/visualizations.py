import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import imutils

from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras

import keras
from keras import utils, callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Dense, Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Input
from keras.models import load_model, Sequential
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam


def plot_training_history(history):
    """
        Utility function for plotting of the model results.

        Args:
            * history: History object returned by tensorflow/keras model after training.
    """

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(train_accuracy))

    plt.figure()

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Training and Validation Accuracies')
    ax1.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
    ax1.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.xlabel('epochs')
    plt.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title('Training and Validation Loss')
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.xlabel('epochs')
    plt.legend()

    plt.show()


def plot_PCA(model, train_generator, test_generator, df_labels):
    """
        Show PCA

        Args:
            * model (keras.Sequential): Trained model object.
    """

    feature_extractor = Model(model.inputs, model.layers[-2].output)  # Dense(128,...)

    train_features = feature_extractor.predict(train_generator)
    test_features = feature_extractor.predict(test_generator)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_train_features = tsne.fit_transform(train_features)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_test_features = tsne.fit_transform(test_features)

    # Split data into train-test data sets
    X = df_labels.loc[:, 'file_name']
    y = df_labels.loc[:, 'tumor_type']
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27, stratify=y)
    train_x, _, train_y, _ = train_test_split(train_x, train_y, test_size=0.1, random_state=27, stratify=train_y)

    # Train df
    df_train = pd.DataFrame({'file_name': train_x, 'tumor_type': train_y})
    df_train.reset_index(drop=True, inplace=True)
    df_train['tumor_type'] = df_train['tumor_type'].astype('str')

    # Test df
    df_test = pd.DataFrame({'file_name': test_x, 'tumor_type': test_y}, columns=['file_name', 'tumor_type'])
    df_test.reset_index(drop=True, inplace=True)
    df_test['tumor_type'] = df_test['tumor_type'].astype('str')

    df_tsne_train = pd.DataFrame({'X': tsne_train_features[:, 0], 'Y': tsne_train_features[:, 1], 'target': df_train['tumor_type'], 'type': 'train'})
    df_tsne_test = pd.DataFrame({'X': tsne_test_features[:, 0], 'Y': tsne_test_features[:, 1], 'target': df_test['tumor_type'], 'type': 'test'})
    df_tsne = pd.concat([df_tsne_train, df_tsne_test])

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.grid(b=None)
    # palette=['dodgerblue','red'],
    sns.color_palette("deep", as_cmap=True, n_colors=4)
    ax = sns.scatterplot(x='X', y='Y', hue='target', legend='full', alpha=1, data=df_tsne_train, palette='deep')
    ax = sns.scatterplot(x='X', y='Y', hue='target', legend='full', alpha=1, data=df_tsne_test, palette='deep')
    handles, labels = ax.get_legend_handles_labels()
    labels_index = {'no_tumor': 0, 'meningioma_tumor': 1, 'glioma_tumor': 2, 'pituitary_tumor': 3}
    ax.legend(handles, labels_index)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.grid(b=None)
    # palette=['dodgerblue','red'],
    # sns.color_palette("deep", as_cmap=True, n_colors=4)
    markers = {"train": "s", "test": "X"}
    ax = sns.scatterplot(x='X', y='Y', hue='target', legend='full', alpha=1, data=df_tsne, palette='deep', style='type', markers=markers)
    handles, labels = ax.get_legend_handles_labels()
    labels_index = {'no_tumor': 0, 'meningioma_tumor': 1, 'glioma_tumor': 2, 'pituitary_tumor': 3}
    ax.legend(handles, labels_index)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()


def plot_datagenerator_sampeles(datagenerator):
    # plotting images
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 6))
    for i in range(5):
        # convert to unsigned integers for plotting
        sample = next(datagenerator)
        image = sample[0].astype('uint8')
        label = sample[1][0]
        labels_index = {0: 'no_tumor', 1: 'meningioma_tumor', 2: 'glioma_tumor', 3: 'pituitary_tumor'}
        # plot raw pixel data
        ax[i].title.set_text(f'{labels_index[label]}')
        ax[i].imshow(image[0])
        ax[i].axis('off')
