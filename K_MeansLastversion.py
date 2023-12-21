#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure, io as skio, color, feature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import logging
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from skimage import io, color, feature, exposure
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Path to the root folder of your dataset
dataset_path = 'D:/fruitsk'

# List all subdirectories (assuming each subdirectory corresponds to a class)
class_folders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
features_list = []
labels_list = []

# Function to extract HOG features from an image and visualize it
def extract_hog_features(image):
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)

    # Calculate HOG features
    hog_features, hog_image = hog(gray_image, visualize=True)

    # Enhance the contrast of the HOG image for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return hog_features, hog_image_rescaled


# In[3]:


# Loop through each class folder
for class_folder in class_folders:
    class_name = os.path.basename(class_folder)

    # Loop through each image in the class folder
    for image_filename in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_filename)

         # Load the image
        image = io.imread(image_path)

        # Extract HOG features and visualize
        hog_features, hog_image = extract_hog_features(image)

        # Display the original image and the HOG features
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 2), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)

        ax2.axis('off')
        ax2.imshow(hog_image, cmap=plt.cm.gray)

        plt.show()

        # Append HOG features to the features list
        features_list.append(hog_features)

        # Append the label to the labels list
        labels_list.append(class_name)


# In[4]:


# Convert lists to NumPy arrays
features_array = np.array(features_list)
labels_array = np.array(labels_list)

# Use LabelEncoder to convert class names into numeric labels
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels_array)

print(features_array.shape)
print(numeric_labels.shape)
print(numeric_labels)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(features_array[:,:1000], numeric_labels, test_size=0.3, random_state=42, stratify=numeric_labels)


# In[6]:


# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(10, 100))

# Fit and transform the scaler on the training data
X_train_normalized = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_normalized = scaler.transform(X_test)

# Colors and labels for clusters
cluster_colors = ['red', 'green', 'black']
cluster_labels = ['avocado', 'Cauliflower', 'tomato']

# Fit KMeans on the normalized training data
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=30)  
kmeans.fit(X_train_normalized)

# Predict cluster labels for the test data
y_pred = kmeans.predict(X_test_normalized)

# Visualize clusters in 2D (using the first two principal components)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_normalized)
X_test_pca = pca.transform(X_test_normalized)

# Plot the test set clusters with colors and labels
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_pred, palette=cluster_colors, style=y_test, markers=["o", "s", "D"], legend='full')
scatter.legend_.set_title('Predicted Labels')  # Corrected legend title

# Add cluster labels
for i, label in enumerate(cluster_labels):
    plt.scatter([], [], color=cluster_colors[i], label=f'Cluster {label}')

plt.title('KMeans Clusters on Test Set')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

accuracy = accuracy_score(y_test, y_pred)

print(f" Accuracy: {accuracy}")

# Binarize the labels
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_pred_bin = label_binarize(y_pred, classes=np.unique(y_test))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(cluster_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'black']
for i, color in zip(range(len(cluster_labels)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve for class {} (area = {:.2f})'.format(cluster_labels[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




