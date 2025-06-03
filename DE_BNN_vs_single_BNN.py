from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN

#%% Train Deep Ensemble of BNNs
def train_deep_ensemble(X_train, y_train, X_val, y_val, input_shape, num_classes, n_models, class_labels=None):
    ensemble_models = []
    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}")
        model = ConvolutionalBNN(input_shape, num_classes, len(X_train), class_labels=class_labels)
        model.compile()
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=64, verbose=1)
        ensemble_models.append(model)
    return ensemble_models

#%% Ensemble Prediction (softmax averaging)
def ensemble_predict_logits(ensemble, X):
    all_logits = [model.predict_logits(X) for model in ensemble]
    return np.stack(all_logits, axis=0)  # shape: (n_models, n_samples, num_classes)

def ensemble_predict_proba(ensemble, X):
    logits = ensemble_predict_logits(ensemble, X)
    probs = tf.nn.softmax(logits, axis=-1).numpy()  # shape: (n_models, n_samples, num_classes)
    return np.mean(probs, axis=0)  # shape: (n_samples, num_classes)

def ensemble_predict_classes(ensemble, X):
    probs = ensemble_predict_proba(ensemble, X)
    return np.argmax(probs, axis=1)

#%% Example
# Load MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.

# Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Split validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

input_shape = (28, 28, 1)
num_classes = 10

# ----- Train a single BNN -----
bnn = ConvolutionalBNN(input_shape, num_classes, len(X_train))
bnn.compile()
bnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=128)

ensemble = train_deep_ensemble(X_train, y_train, X_val, y_val, input_shape, num_classes, n_models=5)

# ----- Compare Accuracies -----
acc_bnn = bnn.evaluate_accuracy(X_test, y_test)
acc_ensemble = accuracy_score(y_test, ensemble_predict_classes(ensemble, X_test))

print(f"\nBNN Accuracy: {acc_bnn:.4f}")
print(f"Ensemble Accuracy: {acc_ensemble:.4f}")

#%% Accuracy Plot
import matplotlib.pyplot as plt

plt.bar(["BNN", "Ensemble"], [acc_bnn, acc_ensemble], color=["skyblue", "orange"])
plt.title("Test Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)
plt.grid(axis="y")
plt.show()

#%% Confusion Matrix for Ensemble and Single
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = ensemble_predict_classes(ensemble, X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format=".0f")
plt.title("Ensemble Confusion Matrix")
plt.show()

y_pred_bnn = bnn.predict_classes(X_test)
cm_bnn = confusion_matrix(y_test, y_pred_bnn)
disp_bnn = ConfusionMatrixDisplay(confusion_matrix=cm_bnn)
disp_bnn.plot(cmap="Purples", values_format=".0f")
plt.title("Confusion Matrix – Single BNN")
plt.grid(False)
plt.show()

#%% Uncertainty Visualization (Entropy of Predictions)
from scipy.stats import entropy
from plots.Custom_plot_style import *

probs = ensemble_predict_proba(ensemble, X_test)
entropy_values = entropy(probs.T)  # shape: (n_samples,)

# Assume entropy_values already computed
plt.figure(figsize=(10, 6))
plt.hist(entropy_values, bins=30, color="orchid", edgecolor="black")

plt.title("Prediction Uncertainty (Entropy)")
plt.xlabel("Entropy")
plt.ylabel("Number of samples")

plt.show()

#%%
# Compute and store ensemble accuracies from 1 to 10
ensemble_accuracies = []
ensemble_models = []

for i in range(5):
    print(f"\nTraining model {i+1}/5")
    model = ConvolutionalBNN(input_shape, num_classes, len(X_train))
    model.compile()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=64)
    ensemble_models.append(model)

    y_pred = ensemble_predict_classes(ensemble_models, X_test)
    acc = accuracy_score(y_test, y_pred)
    ensemble_accuracies.append(acc)

# Plot the results
import matplotlib.pyplot as plt
ensemble_sizes = np.arange(1, 6)

# Include the single model as the first ensemble point
ensemble_accuracies = [acc_bnn] + ensemble_accuracies  # Prepend single model acc
ensemble_sizes = np.arange(1, len(ensemble_accuracies) + 1)

plt.figure(figsize=(8, 5))
plt.plot(ensemble_sizes, ensemble_accuracies, marker='s', label='Ensemble', color='tab:blue')
plt.hlines(acc_bnn, 1, 5, colors='tab:red', linestyles='--', label='Single Model')
plt.xlabel("Ensemble size")
plt.ylabel("Test accuracy")
plt.title("Test Accuracy vs Ensemble Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()