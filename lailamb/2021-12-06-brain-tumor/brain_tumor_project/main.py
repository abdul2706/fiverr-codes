from brain_tumor_project.dataset import *
from brain_tumor_project.baseline_models import *
from brain_tumor_project.main_model import *
from brain_tumor_project.visualizations import *

# load dataset
images, labels = load_dataset()

# create trainset and testset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# preprocess(dataset)

# train baselines
train_KNN()
train_SVM()
# to train KNN on different values of K and then plot the results
# run_KNN_experiment()

model = create_CNN_model(X_train)
model.summary()

visualize_results()

if os.path.exists(BASE_PATH):
    print('Dataset path is set properly.')

df_labels = read_labels(LABELS_PATH)
images = read_images(IMAGES_PATH)

print('*** Top 10 Rows ***')
print(df_labels.head(10))
print('*** Check Number of Rows ***')
print(df_labels.count())

print('images.shape:', images.shape)

generate_brain_contour_crops(IMAGES_PATH)

"""
Main function to start the train and test process.
"""
print('Code starts here: Brain Tumor classification')


# args

# load data

# preprocess data


# run model


if model = 'knn':

elseif
