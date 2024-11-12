
import os  # Provides functions for interacting with the operating system, useful for file and directory handling
import numpy as np  # Imports NumPy, a powerful library for numerical computing with support for arrays and matrices
from skimage.io import imread  # Imports the `imread` function from scikit-image, which allows us to read image files
from skimage.measure import label, regionprops  # Imports functions for image processing; `label` is used for segmentation, 
                                                # and `regionprops` extracts properties from labeled image regions
from sklearn.model_selection import train_test_split  # Imports `train_test_split` to split data into training, validation, and test sets
from sklearn.preprocessing import StandardScaler  # Imports `StandardScaler` for standardizing features to a mean of 0 and standard deviation of 1
from sklearn.neighbors import KNeighborsClassifier  # Imports `KNeighborsClassifier`, a k-Nearest Neighbors algorithm for classification tasks
from sklearn.ensemble import RandomForestClassifier  # Imports `RandomForestClassifier`, a classification algorithm using multiple decision trees
from sklearn.metrics import confusion_matrix, classification_report  # Imports functions to evaluate classification results;
                                                                     # `confusion_matrix` provides a summary of prediction accuracy across classes,
                                                                     # and `classification_report` gives metrics like precision, recall, and F1-score
import matplotlib.pyplot as plt  # Imports Matplotlib's plotting module, useful for creating visualizations, like plotting confusion matrices

# Path to the dataset
DATASET_PATH = "mpeg7_mod"

# Function to load images and their corresponding labels
def loadImagesNLabels(dataset_path):
    """
        This function loads all images from the dataset and extracts features for each image.
        It returns the feature set (X) and the corresponding labels (y) for all images.

        Args:
        - dataset_path (str): The path to the dataset folder.

        Returns:
        - X (ndarray): A list of feature vectors extracted from each image.
        - y (ndarray): A list of labels for each image, corresponding to the folder name.
    """
    X, y = [], []
    # Loop through each folder in the dataset (each folder represents a class)
    for label_folder in os.listdir(dataset_path):
        label_folder_path = os.path.join(dataset_path, label_folder)
        if os.path.isdir(label_folder_path):
            for image_name in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_name)                
                image = imread(image_path, as_gray=True) # Load the image as grayscale                
                X.append(extractFeatures(image)) # Extract features from the image                
                y.append(label_folder) # Label corresponds to the folder name
    return np.array(X), np.array(y)

# Function to extract morphological features from the segmented image
def extractFeatures(image):
    """
        This function extracts morphological features from a binary image.
        The image is segmented by identifying the connected components (objects).
        It then computes several properties (area, perimeter, eccentricity, etc.) of the largest region.

        Args:
        - image (ndarray): The binary image where the shape is white (foreground) and black (background).

        Returns:
        - features (list): A list of morphological features.
    """
    label_image = label(image < 0.5)  # Segment the image by labeling regions (binary threshold at 0.5)
    props = regionprops(label_image)  # Get the properties of labeled regions

    # Extract features for the largest region of interest (the shape)
    if props:
        largest_region = max(props, key=lambda r: r.area)  # Get the region with the largest area
        features = [
            largest_region.area,  # Area of the largest shape
            largest_region.perimeter,  # Perimeter of the shape
            largest_region.eccentricity,  # Eccentricity (measure of elongation)
            largest_region.extent,  # Extent (fraction of the image covered by the region)
            largest_region.solidity,  # Solidity (compactness of the region)
            largest_region.major_axis_length / largest_region.minor_axis_length if largest_region.minor_axis_length > 0 else 0,  # Aspect ratio
            largest_region.orientation  # Orientation (angle of the shape)
        ]
    else:
        features = [0] * 7  # If no region is found, return zeroed features
    return features

# Function to evaluate the classifier's performance and print results
def evaluateClassifier(classifier, X, y, name):
    """
        This function evaluates the classifier on the provided dataset, prints the confusion matrix and classification report,
        and saves the results to a text file.

        Args:
        - classifier (model): A trained machine learning classifier.
        - X (ndarray): The input data to classify.
        - y (ndarray): The true labels for the input data.
        - name (str): The name of the classifier being evaluated.

        Returns:
        - y_pred (ndarray): Predicted labels for the input data.
    """
    y_pred = classifier.predict(X)  # Predict the labels for the input data
    # Create a file and write the evaluation results to it
    with open("results.txt", "a") as f:  # Open in append mode to add results for each classifier
        f.write(f"--- {name} Classifier ---\n")        
        # Write the confusion matrix
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(y, y_pred)) + "\n\n")        
        # Write the classification report
        f.write("Classification Report:\n")
        f.write(classification_report(y, y_pred) + "\n\n")    
    # Also print results to console
    print(f"--- {name} Classifier ---")
    print(confusion_matrix(y, y_pred))  # Print the confusion matrix
    print(classification_report(y, y_pred))  # Print the classification report
    return y_pred

# Function to plot the confusion matrix
def plotConfusionMatrix(y_true, y_pred, title="Confusion Matrix"):
    """
        This function plots the confusion matrix as a heatmap.

        Args:
        - y_true (ndarray): The true labels.
        - y_pred (ndarray): The predicted labels.
        - title (str): The title for the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)  # Compute confusion matrix based on true labels (y_true) and predicted labels (y_pred)
    plt.figure(figsize=(8, 6))  # Set the size of the plot figure to 8x6 inches for better visualization
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Display the confusion matrix as a heatmap using blue color shades
    plt.title(title)  # Set the title 
    plt.colorbar()  # Add a color bar next to the heatmap to indicate the intensity of values
    tick_marks = np.arange(len(set(y_true)))  # Determine tick positions based on the unique classes in y_true
    plt.xticks(tick_marks, set(y_true), rotation=45)  # Set x-axis labels to class names, rotated for readability
    plt.yticks(tick_marks, set(y_true))  # Set y-axis labels to class names (same as x-axis for a confusion matrix)
    plt.xlabel("Predicted Label")  # Label the x-axis as "Predicted Label" to indicate what model predicted
    plt.ylabel("True Label")  # Label the y-axis as "True Label" to indicate the actual true values
    plt.show()  # Display the completed plot

# --- Main procedure ---
# Step 1: Load and segment images, extracting features
X, y = loadImagesNLabels(DATASET_PATH)

# Step 2: Split the dataset into training, validation, and testing sets
# First, split into training+validation (70%) and testing (30%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Then, split the training+validation set into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Step 3: Normalize the feature data
# Fit the scaler only on the training set to avoid data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit the scaler to training data and transform it
X_val = scaler.transform(X_val)  # Apply the same scaling to validation data
X_test = scaler.transform(X_test)  # Apply the same scaling to test data

# Step 4: Train classifiers (k-NN and Random Forest)
# Train k-NN classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Train Random Forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 5: Evaluate classifiers on the test set
# k-NN evaluation
print("Evaluating k-NN classifier:")
knn_pred = evaluateClassifier(knn, X_test, y_test, "k-NN")
# Random Forest evaluation
print("Evaluating Random Forest classifier:")
rf_pred = evaluateClassifier(rf, X_test, y_test, "Random Forest")
# Plot confusion matrix for both classifiers
plotConfusionMatrix(y_test, knn_pred, title="Confusion Matrix - k-NN")
plotConfusionMatrix(y_test, rf_pred, title="Confusion Matrix - Random Forest")
