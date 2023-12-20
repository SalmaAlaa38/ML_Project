#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure, io as skio, color, feature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import os
import seaborn as sns
from sklearn.utils import shuffle
from itertools import cycle
from skimage import io, color, feature, exposure


# In[2]:


# Path to the root folder of your dataset
dataset_path = 'D:/fruitss'

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


# In[12]:


# Split the dataset into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(features_array[:, :700], numeric_labels, test_size=0.5,
                                                    random_state=44, stratify=numeric_labels)


# In[13]:


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the scaler on the training data
X_train_normalized = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_normalized = scaler.transform(X_test)

# Initialize the logistic regression model
logistic_model = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, random_state=44, warm_start=True, max_iter=1000)

# Train the model on the normalized training data
logistic_model.fit(X_train_normalized, y_train)

# Make predictions on the normalized test set
y_pred = logistic_model.predict(X_test_normalized)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred, zero_division=1)
print(classification_report_result)

# Print the results
print(f'Accuracy: {accuracy}')
CM = confusion_matrix(y_test, y_pred)
sns.heatmap(CM, annot=True, fmt="d", cmap="Blues", center=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Compute ROC curve and ROC area for each class
y_score = logistic_model.predict_proba(X_test_normalized)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(np.unique(y_test))):
    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'limegreen', 'mediumorchid', 'gold', 'firebrick'])

for i, color in zip(range(len(np.unique(y_test))), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve (class {label_encoder.classes_[i]}) (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




