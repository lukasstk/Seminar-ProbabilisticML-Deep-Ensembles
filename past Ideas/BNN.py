import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tf_keras
import tensorflow_probability as tfp

tfd = tfp.distributions

# Load data
red_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
red = pd.read_csv(red_url, sep=';')

# Binary target: quality ≥ 6 → 1, else 0
X = red.drop('quality', axis=1).astype(np.float32)
y = (red['quality'] >= 6).astype(np.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define KL divergence function (scaled by dataset size)
kl_divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / X_train.shape[0]

# Define the BNN model with Flipout layers
model = tf_keras.Sequential([
    tfp.layers.DenseFlipout(32, activation='relu', kernel_divergence_fn=kl_divergence_fn, input_shape=(11,)),
    tfp.layers.DenseFlipout(16, activation='relu', kernel_divergence_fn=kl_divergence_fn),
    tfp.layers.DenseFlipout(1, activation='sigmoid', kernel_divergence_fn=kl_divergence_fn)
])

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)
