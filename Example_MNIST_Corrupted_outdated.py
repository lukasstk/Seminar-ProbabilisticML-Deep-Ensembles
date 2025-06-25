from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN
from tensorflow_datasets.image_classification import MNISTCorrupted
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import train_test_split
from DE_BNN_vs_single_BNN import *
from plots.Plots_Helper import save_plots

# === Schritt 1: impulse_noise laden (nur test split verfügbar) ===
builder = MNISTCorrupted(config="impulse_noise")
builder.download_and_prepare()

ds = builder.as_dataset(split="test", as_supervised=True)
X_data, y_data = zip(*[(x, y) for x, y in tfds.as_numpy(ds)])

X_data = np.array(X_data)[..., np.newaxis] / 255.0
y_data = np.array(y_data)

# === Schritt 2: Split in Train/Test (da nur test-Split vorhanden ist) ===
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42)

# === Schritt 3: Modell trainieren und evaluieren ===
model_single = ConvolutionalBNN(input_shape=(28, 28, 1), num_classes=10, len_x_train=len(X_train))
model_single.compile()
model_single.fit(X_train, y_train, epochs=5, batch_size=128, verbose=1, validation_split=0.1)

acc_single = model_single.evaluate_accuracy(X_test, y_test)
print(f"BNN Single Accuracy (impulse_noise): {acc_single:.4f}")

preds = model_single.predict_classes(X_test[:10])
print("Predicted labels:", preds)

#%% Vergleich zw. DE und single
from tensorflow_datasets.image_classification import MNISTCorrupted
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
from Ensemble_helper import train_deep_ensemble, evaluate_model
from plots.Plots_main import plot_ensemble_metrics
import pandas as pd
"""# ───────────────────────────────────────────────────────────────
# Verfügbare Konfigurationen (Korruptionsarten) für MNISTCorrupted:
# Quelle: https://www.tensorflow.org/datasets/catalog/mnist_corrupted
# Erwarteter Drop gegenüber cleanem MNIST (≈98% Accuracy):
#
# 1.  'gaussian_noise'        → leicht       (↓ 5–10%)
# 2.  'shot_noise'            → mittel       (↓ 10–20%)
# 3.  'impulse_noise'         → stark        (↓ 60–80%)
# 4.  'defocus_blur'          → mittel       (↓ 15–25%)
# 5.  'glass_blur'            → stark        (↓ 30–50%)
# 6.  'motion_blur'           → mittel       (↓ 10–25%)
# 7.  'zoom_blur'             → leicht       (↓ 5–10%)
# 8.  'snow'                  → mittel       (↓ 15–25%)
# 9.  'frost'                 → mittel       (↓ 10–20%)
# 10. 'fog'                   → mittel       (↓ 10–20%)
# 11. 'brightness'            → leicht       (↓ 5–10%)
# 12. 'contrast'              → leicht       (↓ 5–10%)
# 13. 'elastic_transform'     → leicht/mittel(↓ 5–15%)
# 14. 'pixelate'              → mittel       (↓ 15–30%)
# 15. 'jpeg_compression'      → gering       (↓ 0–5%)
# 16. 'rotate'                → stark        (↓ 40–60%)
# 17. 'shear'                 → mittel       (↓ 10–25%)
# 18. 'translate'             → mittel       (↓ 15–30%)
# 19. 'scale'                 → mittel       (↓ 15–30%)"""

# === Load impulse_noise corrupted MNIST ===
builder = MNISTCorrupted(config="impulse_noise")
builder.download_and_prepare()

ds = builder.as_dataset(split="test", as_supervised=True)
X_all, y_all = zip(*[(x, y) for x, y in tfds.as_numpy(ds)])
X_all = np.array(X_all) / 255.0
y_all = np.array(y_all)

# === Split into train/val/test (e.g., 60/20/20 split) ===
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# === Train new single BNN on corrupted data ===
bnn_corr = ConvolutionalBNN(input_shape=(28, 28, 1), num_classes=10, len_x_train=len(X_train))
bnn_corr.compile()
bnn_corr.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=128)

# === Train new DE-BNNs on corrupted data ===
ensemble_corr = train_deep_ensemble(X_train, y_train, X_val, y_val,
                                     input_shape=(28, 28, 1), num_classes=10, n_models=5)

#%% Evaluate both models
print("\nEvaluating DE-BNN")
y_proba_de = ensemble_predict_proba(ensemble_corr, X_test)
results_de = evaluate_model(y_test, y_proba_de)

#%% Print comparison
results_df = pd.DataFrame({
    """"Single-BNN": results_single,"""
    "DE-BNN": results_de
})
print("\nComparison of Evaluation Metrics:")
print(results_df)

# 1. Get predicted probabilities from each ensemble model
probs_list = [model.predict_proba(X_test) for model in ensemble]

# 2. Evaluate for each ensemble size (1 to 5)
results_per_size = []

for k in range(1, len(ensemble) + 1):
    probs_k = np.mean(probs_list[:k], axis=0)  # Average predictions of first k models
    metrics = evaluate_model(y_test, probs_k, num_classes=10)
    metrics["Ensemble Size"] = k
    results_per_size.append(metrics)

# 3. Convert to DataFrame for analysis or plotting
df_ensemble_metrics = pd.DataFrame(results_per_size)
print(df_ensemble_metrics)

plots = plot_ensemble_metrics(df_ensemble_metrics, df_ensemble_metrics.iloc[0,:], mnist=False)

save_plots(plots, file_suffix="Corrupted")