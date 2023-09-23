import argparse
from sklearn.model_selection import train_test_split

# import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

from brain_tumor_project.utils import *
from brain_tumor_project.models import *
from brain_tumor_project.dataset import *

"""
Main function to start the train and test process.
"""
print('Code starts here: Brain Tumor classification')

# args
parser = argparse.ArgumentParser(description='Brain Tumor Classification | Training, Testing and Visualization')
parser.add_argument('--base_path', type=str, default='./brain-tumor-dataset', help='Base path of dataset')
parser.add_argument('--epochs', type=int, default=5, help='Number of Epochs for training all models')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training all models')
parser.add_argument('--save_models_dir', type=int, default=100, help='Directory path for saving trained models')

args = parser.parse_args()
print(args)

BASE_PATH = args.base_path
IMAGES_PATH = os.path.join(BASE_PATH, 'images')
CROP_IMAGES_PATH = os.path.join(BASE_PATH, 'crop-images')
LABELS_PATH = os.path.join(BASE_PATH, 'labels.csv')
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
SAVE_MODELS_PATH = './saved_models'
if not os.path.exists(SAVE_MODELS_PATH):
    os.mkdir(SAVE_MODELS_PATH)
EPOCHS = 1

########################################################
# Load Data
images = read_images(IMAGES_PATH)
print('images.shape:', images.shape)
df_labels = read_labels(LABELS_PATH)
print('*** Top 10 Rows ***')
print(df_labels.head(10))
print('*** Check Number of Rows ***')
print(df_labels.count())
generate_brain_contour_crops(IMAGES_PATH, CROP_IMAGES_PATH)
########################################################


########################################################
# Data Visualizations
plot_samples(images, df_labels, 5)
plot_hist(df_labels, 'tumor_type', 'Tumor Types')
plot_hist(df_labels, 'tumor', 'Tumor/No-Tumor')
########################################################


########################################################
# Preprocess Data
# create training and test sets
X_train_tumor, X_test_tumor, y_train_tumor, y_test_tumor = train_test_split(images, df_labels.tumor, test_size=0.2, random_state=42, stratify=df_labels.tumor)
X_train_tumor_type, X_test_tumor_type, y_train_tumor_type, y_test_tumor_type = train_test_split(images, df_labels.tumor_type, test_size=0.2, random_state=42, stratify=df_labels.tumor_type)

# create train, val and test datagenerators
train_generator_tumor, val_generator_tumor, test_generator_tumor = get_data_generators(IMAGES_PATH, df_labels, 'file_name', 'tumor', 'binary', target_size=(100, 100))
plot_datagenerator_sampeles(train_generator_tumor)
train_generator_tumor_type, val_generator_tumor_type, test_generator_tumor_type = get_data_generators(IMAGES_PATH, df_labels, 'file_name', 'tumor_type', 'sparse', target_size=(150, 150))
plot_datagenerator_sampeles(train_generator_tumor_type)
########################################################


########################################################
# Train, Test, Evaluate Models
# train and test baseline models
print('Calling train_test_SVC')
train_test_SVC(X_train_tumor, y_train_tumor, X_test_tumor, y_test_tumor)
print('Calling KNN_neighbor_search')
KNN_neighbor_search(X_train_tumor, y_train_tumor, X_test_tumor, y_test_tumor)
print('Calling train_test_KNN')
train_test_KNN(X_train_tumor, y_train_tumor, X_test_tumor, y_test_tumor)
########################################################


########################################################
# train, test and visialize results of various CNN models
############################
# 3.1 CNN (from scratch)
model = CNN_model(input_shape=(100, 100, 3))
model.summary()
history = train_CNN(model, X_train_tumor, y_train_tumor, val_ratio=0.1,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, metrics=['accuracy'],
                    loss='binary_crossentropy', optimizer='adam',
                    class_weights_flag=None, verbose=1)
plot_training_history(history)
test_CNN(model, X_test_tumor, y_test_tumor, target_names=['No tumor', 'Tumor'])
model.save(os.path.join(SAVE_MODELS_PATH, 'CNN-from-scratch.h5'))
print('Saved model to disk: CNN-from-scratch.h5')
############################

############################
# 3.2 Fine-tuning
model = VGG16_model1(input_shape=(100, 100, 3))
model.summary()
history = train_CNN(model, X_train_tumor, y_train_tumor, val_ratio=0.1,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, metrics=['accuracy'],
                    loss='binary_crossentropy', optimizer='adam',
                    class_weights_flag=None, verbose=1)
plot_training_history(history)
test_CNN(model, X_test_tumor, y_test_tumor, target_names=['No tumor', 'Tumor'])
model.save(os.path.join(SAVE_MODELS_PATH, 'VGG16-fine-tuning.h5'))
print('Saved model to disk: VGG16-fine-tuning.h5')
############################

############################
# 3.3 Fine-tuning with data augmentation
model = VGG16_model1(input_shape=(100, 100, 3))
model.summary()
history = train_CNN_using_generator(model, train_generator_tumor, val_generator_tumor, y_train_tumor,
                                    epochs=EPOCHS, batch_size=BATCH_SIZE, metrics=['accuracy'],
                                    loss='binary_crossentropy', optimizer='adam', verbose=1)
plot_training_history(history)
test_CNN(model, test_generator_tumor, y_test_tumor, target_names=['No tumor', 'Tumor'])
model.save(os.path.join(SAVE_MODELS_PATH, 'VGG16-fine-tuning-with-augmentation.h5'))
print('Saved model to disk: VGG16-fine-tuning-with-augmentation.h5')
############################

############################
# 4.1 Multi-label
model = VGG16_model2(input_shape=(150, 150, 3))
model.summary()
history = train_CNN_using_generator(model, train_generator_tumor_type, val_generator_tumor_type, y_train_tumor,
                                    epochs=EPOCHS, batch_size=BATCH_SIZE, metrics=['accuracy'],
                                    loss='sparse_categorical_crossentropy', optimizer='adam',
                                    class_weights_flag=True, callbacks_flag=True, verbose=1)
plot_training_history(history)
test_CNN(model, test_generator_tumor_type, y_test_tumor_type, target_names=['No tumor', 'meningioma', 'glioma', 'pituitary'])
model.save(os.path.join(SAVE_MODELS_PATH, 'VGG16-multi-label.h5'))
print('Saved model to disk: VGG16-multi-label.h5')

fine_tuning_flag = True
if fine_tuning_flag:
    initial_learning_rate = 1e-5
    lr_schedule = ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    EPOCHS = 1
    optimizer = Adam(learning_rate=lr_schedule)
    history = train_CNN_using_generator(model, train_generator_tumor_type, val_generator_tumor_type, y_train_tumor,
                                        epochs=EPOCHS, batch_size=BATCH_SIZE, metrics=['accuracy'],
                                        loss='sparse_categorical_crossentropy', optimizer=optimizer,
                                        class_weights_flag=True, callbacks_flag=True, verbose=1)
    plot_training_history(history)
    test_CNN(model, test_generator_tumor_type, y_test_tumor_type, target_names=['No tumor', 'meningioma', 'glioma', 'pituitary'])
    model.save(os.path.join(SAVE_MODELS_PATH, 'VGG16-multi-label-fine-tuned.h5'))
    print('Saved model to disk: VGG16-multi-label-fine-tuned.h5')
############################
########################################################


########################################################
# PCA Visualization
MODEL_TO_LOAD = 'VGG16-multi-label-fine-tuned.h5'
model = keras.models.load_model(os.path.join(SAVE_MODELS_PATH, MODEL_TO_LOAD))
plot_PCA(model, train_generator_tumor_type, test_generator_tumor_type, df_labels)
########################################################
