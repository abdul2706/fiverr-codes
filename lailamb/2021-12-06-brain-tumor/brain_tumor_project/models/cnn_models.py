from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, Flatten, Input
from keras.layers.advanced_activations import LeakyReLU


def CNN_model(input_shape):
    # CNN (from scratch)
    model = Sequential()
    # level 1
    model.add(Conv2D(64, (5, 5), input_shape=input_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # level 2
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # level 3
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # flatten all neurons
    model.add(Flatten())
    # linear layers
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def VGG16_model1(input_shape=(100, 100, 3)):
    # creating the base model of pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze all the layers
    for layer in base_model.layers[:]:
        layer.trainable = False
    # build a classifier model to put on top of the convolutional model
    model = Sequential()
    model.add(base_model)
    # flatten all neurons
    model.add(Flatten(input_shape=base_model.output_shape[1:]))
    # linear layers level 1
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # linear layers level 2
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # output layer
    model.add(Dense(1, activation='sigmoid'))
    return model


def VGG16_model2(input_shape=(150, 150, 3)):
    # creating the base model of pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze all the layers
    for layer in base_model.layers[:]:
        layer.trainable = False
    # build a classifier model to put on top of the convolutional model
    model = Sequential()
    model.add(base_model)
    # flatten all neurons
    # model.add(GlobalAveragePooling2D())  # try GlobalAveragePooling2D instead of Flatten
    model.add(Flatten(input_shape=base_model.output_shape[1:]))
    # linear layers level 1
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # linear layers level 2
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # output layer
    model.add(Dense(4, activation='softmax'))
    return model
