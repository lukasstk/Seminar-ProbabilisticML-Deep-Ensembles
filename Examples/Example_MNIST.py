import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Model_Code.ConvolutionalBNN_Model import ConvolutionalBNN

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
