#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# Function to perform gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        
        # Avoid log(0) and log(1) by adding a small epsilon
        epsilon = 1e-10
        cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient
        cost_history.append(cost)

    return theta, cost_history

# Function to calculate the logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Path to the root folder of your dataset
dataset_path = 'D:/fruitss'

# List all subdirectories (assuming each subdirectory corresponds to a class)
class_folders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
features_list = []
labels_list = []

# Loop through each class folder
for class_folder in class_folders:
    class_name = os.path.basename(class_folder)

    # Loop through each image in the class folder
    for image_filename in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_filename)

        # Load the image
        image = io.imread(image_path)

        # Convert the image to grayscale
        gray_image = color.rgb2gray(image)

        # Extract HOG features and visualize
        hog_features, hog_image = hog(gray_image, visualize=True)

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

# Convert lists to NumPy arrays
features_array = np.array(features_list)
labels_array = np.array(labels_list)

# Use LabelEncoder to convert class names into numeric labels
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels_array)

# Add a bias term to the features
X_train = np.c_[np.ones((len(features_array), 1)), features_array]


# Split the dataset into training and testing sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(features_array[:,:700], numeric_labels, test_size=0.50,
                                                    random_state=44, stratify=numeric_labels)

# Initialize theta with zeros
theta = np.zeros(X_train.shape[1])

# Set hyperparameters for gradient descent
learning_rate = 0.01
iterations = 500

# Perform gradient descent
theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# Plot the cost history to visualize convergence
plt.plot(range(1, iterations + 1), cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of Gradient Descent')
plt.show()

# Make predictions on the test set using logistic regression
z = np.dot(X_test, theta)
y_pred_logistic = sigmoid(z)
y_pred_binary_logistic = (y_pred_logistic >= 0.5).astype(int)

# Evaluate the logistic regression model
accuracy_logistic = accuracy_score(y_test, y_pred_binary_logistic)
classification_report_result_logistic = classification_report(y_test, y_pred_binary_logistic, zero_division=1)

print(f"Logistic Regression Accuracy: {accuracy_logistic}")
print("Logistic Regression Classification Report:")
print(classification_report_result_logistic)

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
print(f"Logistic Regression with StandardScaler Accuracy: {accuracy}")
print("Logistic Regression with StandardScaler Classification Report:")
print(classification_report_result)

# Print the results
print(f'Accuracy: {accuracy}')
CM = confusion_matrix(y_test, y_pred)
sns.heatmap(CM, annot=True, fmt="d", cmap="Blues", center=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Compute ROC curve and ROC area for each class using logistic regression with StandardScaler
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
plt.title('Receiver Operating Characteristic (ROC) Curve - Logistic Regression with StandardScaler')
plt.legend(loc="lower right")
plt.show()

