from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, brier_score_loss

# Load CIFAR-10
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
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
    bnn = ConvolutionalBNN(input_shape=input_shape, num_classes=num_classes, len_x_train=len_x_train)
    bnn.compile()
    bnn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64, verbose=0)
    models.append(bnn)

    # Predict logits on test set and convert to probabilities
    logits = bnn.predict_logits(x_test)
    probs = tf.nn.softmax(logits).numpy()
    probs_list.append(probs)

probs_list = np.array(probs_list)  # (ensemble_size, N, num_classes)

# Evaluate single model (first)
probs_single = probs_list[0]
preds_single = np.argmax(probs_single, axis=1)
acc_single = accuracy_score(y_test, preds_single)
brier_single = np.mean([
    brier_score_loss((y_test == i).astype(int), probs_single[:, i])
    for i in range(probs_single.shape[1])
])

# Evaluate ensembles of size 1 to ensemble_size
ensemble_sizes = list(range(1, ensemble_size + 1))
accs = []
briers = []

for k in ensemble_sizes:
    partial_probs = np.mean(probs_list[:k], axis=0)
    preds = np.argmax(partial_probs, axis=1)
    accs.append(accuracy_score(y_test, preds))
    brier = np.mean([
        brier_score_loss((y_test == i).astype(int), partial_probs[:, i])
        for i in range(partial_probs.shape[1])
    ])
    briers.append(brier)

# Plot results
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(ensemble_sizes, accs, label="Ensemble", marker='o')
plt.axhline(y=acc_single, color='r', linestyle='--', label="Single")
plt.xlabel("Ensemble size")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Ensemble Size")
plt.legend()

# Brier score plot
plt.subplot(1, 2, 2)
plt.plot(ensemble_sizes, briers, label="Ensemble", marker='o')
plt.axhline(y=brier_single, color='r', linestyle='--', label="Single")
plt.xlabel("Ensemble size")
plt.ylabel("Brier Score")
plt.title("Brier Score vs Ensemble Size")
plt.legend()

plt.tight_layout()
plt.suptitle("ConvolutionalBNN Ensemble on CIFAR-10", fontsize=14, y=1.03)
plt.show()
