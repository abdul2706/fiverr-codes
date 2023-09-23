import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist(df_labels, label_column, title=''):
    """
    This function is to plot a histogram for the specified column in the labels dataframe
    Args:
        * data_frame (data_frame): pandas dataframe of the labels.
        * x (string): the desired column name to plot.
    """

    plt.figure()
    # To change the color visit: https://seaborn.pydata.org/tutorial/color_palettes.html?highlight=color
    # sns.color_palette('bright')
    xtick_labels = list(df_labels[label_column].unique())
    if label_column == 'tumor':
        xtick_labels = ['No-Tumor', 'Tumor']
    elif label_column == 'tumor_type':
        xtick_labels = ['No-Tumor', 'Meningioma-Tumor', 'Glioma-Tumor', 'Pituitary-Tumor']
    sns.countplot(x=label_column,  data=df_labels)
    plt.xticks(range(len(xtick_labels)), xtick_labels)
    plt.xlabel('')
    plt.title(title)
    plt.show()


def plot_samples(imgs, df_labels, num_of_samples=6):
    """
    This function is to plot samples from the dataset
    Args:
        * imgs (ndarray): images array to plot samples from.
        * labels (ndarray or list): true class label of images.
        * num_of_samples (int): number of images/samples to plot.
    """

    plt.figure(figsize=(10, 6))
    cols = min(7, num_of_samples)
    rows = num_of_samples // cols + (1 if num_of_samples > 7 else 0)
    for i in range(num_of_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(df_labels.loc[i, 'label'])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
