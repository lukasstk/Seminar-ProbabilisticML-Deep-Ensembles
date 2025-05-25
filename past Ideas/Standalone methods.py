import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dropout
import random
import tf_keras as tfk
from sklearn.utils import shuffle


# --- Standalone method 1: get logits from model ---
def predict_logits(model, X_test):
    """
    Returns raw logits (before softmax) for X_test.
    Shape: (num_samples, num_classes)
    """
    return model(X_test, training=False).numpy()


# --- Standalone method 2: get predicted classes from logits ---
def predict_classes(model, X_test, label_encoder=None, class_labels=None):
    """
    Returns predicted class labels (not indices).
    Handles non-zero-indexed or string class labels if provided.
    """
    logits = predict_logits(model, X_test)
    class_indices = np.argmax(logits, axis=1) # gives the most suitable class for the logits
                                                # (10 elements) --> highest gets chosen --> returns this class

    if label_encoder is not None:
        return label_encoder.inverse_transform(class_indices)  # return strings
    else:
        return [class_labels[i] for i in class_indices]


# --- Standalone method 3: compute accuracy ---
def evaluate_accuracy(model, X_test, y_test, label_encoder=None, class_labels=None):
    """
    Computes accuracy using true labels and predicted labels.
    Assumes y_test contains actual class labels (not necessarily zero-indexed).
    """
    y_pred = predict_classes(model, X_test, label_encoder, class_labels)

    if label_encoder is not None and np.issubdtype(y_test.dtype, np.integer):
        y_test = label_encoder.inverse_transform(y_test)

    return accuracy_score(y_test, y_pred)

class CustomLabelEncoder:
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = []

    def fit(self, labels):
        seen = {}
        for label in labels:
            if label not in seen:
                seen[label] = len(seen)
                self.index_to_label.append(label)
        self.label_to_index = seen
        return self

    def transform(self, labels):
        return np.array([self.label_to_index[label] for label in labels])

    def inverse_transform(self, indices):
        return np.array([self.index_to_label[i] for i in indices])

    def classes_(self):
        return self.index_to_label




# Load and normalize MNIST
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., tf.newaxis] / 255.0
X_test = X_test[..., tf.newaxis] / 255.0

# Convert integer labels to string labels (e.g. 0 → "zero")
class_labels = ["zero", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine"]
y_train_full_str = np.array([class_labels[i] for i in y_train_full])
y_test_str = np.array([class_labels[i] for i in y_test])

# Split train into train/val
from sklearn.utils import shuffle
X_train_full, y_train_full_str = shuffle(X_train_full, y_train_full_str, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full_str, test_size=0.2, random_state=42)

label_encoder = CustomLabelEncoder()
label_encoder.fit(class_labels)  # fits "zero" through "nine"

y_train_encoded = label_encoder.transform(y_train)

