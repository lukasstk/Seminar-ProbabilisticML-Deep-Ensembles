#%% Dataset with Integer Labels
from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN

random_state = np.random.RandomState(42)
# Load and normalize
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., tf.newaxis] / 255.0
X_test = X_test[..., tf.newaxis] / 255.0

# Split full training set into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42)

# Single CNN-BNN Model
model_single = ConvolutionalBNN(input_shape=(28, 28, 1), num_classes=10, len_x_train=len(X_train))
model_single.compile()
model_single.fit(X_train, y_train, epochs=2, batch_size=128, verbose=1, validation_split=0.1)
acc_single = model_single.evaluate_accuracy(X_test, y_test)

print(f"BNN Single Accuracy: {acc_single:.4f}")

model_single_predictions = model_single.predict_classes(X_test[:10]) # predicts integers
print("Predicted integer labels:", model_single_predictions)

#%% Fake class labels: animal names
animal_labels = [
    "cat", "dog", "elephant", "tiger", "lion",
    "giraffe", "zebra", "bear", "wolf", "owl"
]

# Load and preprocess MNIST
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_full = x_train_full[..., tf.newaxis] / 255.0
x_test = x_test[..., tf.newaxis] / 255.0

# Split into train/val
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Build and train BNN
bnn = ConvolutionalBNN(
    input_shape=(28, 28, 1), class_labels=animal_labels, num_classes=10
)
bnn.compile()
bnn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, batch_size=128)

# Evaluate model
accuracy = bnn.evaluate_accuracy(x_test, y_test)
print("Test Accuracy:", accuracy)

# Predict and show animal labels for a few digits
sample_images = x_test[:10]
predicted_animals = bnn.predict_classes(sample_images)
print("Predicted animal labels:", predicted_animals)

#%% old Code without class
"""import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dropout
import random
from tensorflow.keras import datasets
import tf_keras as tfk
from sklearn.utils import shuffle

def kl_divergence_fn(q, p, _):
    return tfp.distributions.kl_divergence(q, p) / x_train.shape[0]

tfpl = tfp.layers

# Build the Bayesian CNN
bnn = tfk.Sequential()

# First conv block
bnn.add(tfpl.Convolution2DFlipout(32, (3, 3), activation='relu', input_shape=(28, 28, 1),
                                  kernel_divergence_fn=kl_divergence_fn))
bnn.add(tfk.layers.MaxPooling2D(pool_size=(2, 2)))

# Second conv block
bnn.add(tfpl.Convolution2DFlipout(64, (3, 3), activation='relu',
                                  kernel_divergence_fn=kl_divergence_fn))
bnn.add(tfk.layers.MaxPooling2D(pool_size=(2, 2)))

# Third conv block
bnn.add(tfpl.Convolution2DFlipout(128, (3, 3), activation='relu',
                                  kernel_divergence_fn=kl_divergence_fn))
bnn.add(tfk.layers.MaxPooling2D(pool_size=(2, 2)))

# Dense layers
bnn.add(tfk.layers.Flatten())
bnn.add(tfpl.DenseFlipout(256, activation='relu', kernel_divergence_fn=kl_divergence_fn))
bnn.add(tfk.layers.Dropout(0.3))
bnn.add(tfpl.DenseFlipout(128, activation='relu', kernel_divergence_fn=kl_divergence_fn))
bnn.add(tfk.layers.Dropout(0.2))

# Output layer (logits)
bnn.add(tfpl.DenseFlipout(10,  kernel_divergence_fn=kl_divergence_fn))  # no softmax, use from_logits=True

bnn.summary()

bnn.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# Load and normalize MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., tf.newaxis] / 255.0
x_test = x_test[..., tf.newaxis] / 255.0

# Train the model
bnn.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)"""
