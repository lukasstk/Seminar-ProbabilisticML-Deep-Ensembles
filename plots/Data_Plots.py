from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN

import numpy as np


# Load MNIST
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Normalize
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Split training into train and validation (e.g., 90% train, 10% val)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)

# Dataset info
input_shape = x_train.shape[1:]               # (32, 32, 3)
num_classes = len(np.unique(y_train))         # 10
len_x_train = len(x_train)

# Ensemble settings
ensemble_size = 5
models = []
probs_list = []

# Train ensemble
for i in range(ensemble_size):
    print(f"Training model {i+1}/{ensemble_size}")
    bnn = ConvolutionalBNN(input_shape=(28,28,1), num_classes=num_classes, len_x_train=len_x_train)
    bnn.compile()
    bnn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=64, verbose=1)
    models.append(bnn)

    # Predict logits on test set and convert to probabilities
    logits = bnn.predict_logits(x_test)
    probs = tf.nn.softmax(logits).numpy()
    probs_list.append(probs)

probs_list = np.array(probs_list)  # (ensemble_size, N, num_classes)