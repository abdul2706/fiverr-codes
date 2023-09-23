import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from keras.preprocessing.image import ImageDataGenerator


def get_data_generators(images_path, df_labels, X_column, y_column, class_mode, target_size=(100, 100)):
    # Split data into train-test data sets
    X = df_labels.loc[:, X_column]
    y = df_labels.loc[:, y_column]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=27, stratify=y)

    # Train df
    df_train = pd.DataFrame({X_column: train_x, y_column: train_y})
    df_train.reset_index(drop=True, inplace=True)
    df_train[y_column] = df_train[y_column].astype('str')

    # Test df
    df_test = pd.DataFrame({X_column: test_x, y_column: test_y}, columns=[X_column, y_column])
    df_test.reset_index(drop=True, inplace=True)
    df_test[y_column] = df_test[y_column].astype('str')

    # Train and Val Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        brightness_range=[0.2, 0.5],
        validation_split=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                        directory=images_path,
                                                        x_col=X_column,
                                                        y_col=y_column,
                                                        class_mode=class_mode,
                                                        target_size=target_size,
                                                        batch_size=100,
                                                        rescale=1.0/255,
                                                        subset='training',
                                                        seed=2020,
                                                        shuffle=False)

    valid_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                        directory=images_path,
                                                        x_col=X_column,
                                                        y_col=y_column,
                                                        class_mode=class_mode,
                                                        target_size=target_size,
                                                        batch_size=100,
                                                        rescale=1.0/255,
                                                        subset='validation',
                                                        seed=2020,
                                                        shuffle=False)

    test_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )

    test_generator = test_datagen.flow_from_dataframe(dataframe=df_test,
                                                      directory=images_path,
                                                      x_col=X_column,
                                                      y_col=y_column,
                                                      class_mode=class_mode,
                                                      target_size=target_size,
                                                      batch_size=100,
                                                      rescale=1.0/255,
                                                      seed=2020,
                                                      shuffle=False)

    return train_generator, valid_generator, test_generator
