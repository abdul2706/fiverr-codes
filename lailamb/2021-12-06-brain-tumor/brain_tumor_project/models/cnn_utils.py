import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import tensorflow as tf
from keras import callbacks


def compute_class_weights(y_train):
    # define class weights
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    computed_class_weights = dict(zip(np.unique(y_train), class_weights))
    print('computed_class_weights:', computed_class_weights)


def train_CNN(model, X_train, y_train, val_ratio=0.1,
              epochs=30, batch_size=100, metrics=['accuracy'],
              loss='binary_crossentropy', optimizer='adam',
              class_weights_flag=False, callbacks_flag=False, verbose=1):

    # compile the model
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    # instantiate callbacks
    earlystopping = callbacks.EarlyStopping(monitor="loss", mode="min", patience=2, restore_best_weights=True)

    # create training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42, stratify=y_train)

    # fit the model
    with tf.device('/gpu:0'):
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[earlystopping] if callbacks_flag else None,
                            class_weight=compute_class_weights(y_train) if class_weights_flag else None,
                            verbose=verbose)

    return history


def train_CNN_using_generator(model, train_generator, val_generator, y_train,
                              epochs=30, batch_size=100, metrics=['accuracy'],
                              loss='binary_crossentropy', optimizer='adam',
                              class_weights_flag=False, callbacks_flag=False, verbose=1):

    # compile the model
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    # instantiate callbacks
    earlystopping = callbacks.EarlyStopping(monitor="loss", mode="min", patience=2, restore_best_weights=True)

    # fit the model
    with tf.device('/gpu:0'):
        history = model.fit(train_generator,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=val_generator,
                            callbacks=[earlystopping] if callbacks_flag else None,
                            class_weight=compute_class_weights(y_train) if class_weights_flag else None,
                            verbose=verbose)

    return history


def test_CNN(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

    # target_names = ['No tumor', 'Tumor']
    print('******* Classification Report *******')
    print(classification_report(y_test, y_pred.round(), target_names=target_names, digits=4))

    # Not Normalized
    print('******* Un-Normalized Confusion Matrix *******')
    cm = confusion_matrix(y_test, y_pred.round())
    print(cm)

    # Normalized
    print('******* Normalized Confusion Matrix *******')
    cm = confusion_matrix(y_test, y_pred.round(), normalize='true')
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp = disp.plot(cmap=plt.cm.Blues)
    plt.show()
