import random
from tensorflow_datasets.image_classification import MNISTCorrupted
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
from Model_Code.ConvolutionalBNN_Model import ConvolutionalBNN
from Model_Code.Ensemble_helper import train_deep_ensemble, evaluate_model, ensemble_predict_proba
from plots.Plots_Helper import plot_ensemble_metrics, save_plots
import pandas as pd
import tensorflow as tf

# ───────────────────────────────────────────────────────────────
# Available configurations (corruption types) for MNIST-Corrupted:
# Source: https://www.tensorflow.org/datasets/catalog/mnist_corrupted
# Expected accuracy drop compared to clean MNIST (≈98% accuracy):
#
# 1.  'gaussian_noise'        → mild         (↓ 5–10%)
# 2.  'shot_noise'            → moderate     (↓ 10–20%)
# 3.  'impulse_noise'         → severe       (↓ 60–80%)
# 4.  'defocus_blur'          → moderate     (↓ 15–25%)
# 5.  'glass_blur'            → severe       (↓ 30–50%)
# 6.  'motion_blur'           → moderate     (↓ 10–25%)
# 7.  'zoom_blur'             → mild         (↓ 5–10%)
# 8.  'snow'                  → moderate     (↓ 15–25%)
# 9.  'frost'                 → moderate     (↓ 10–20%)
# 10. 'fog'                   → moderate     (↓ 10–20%)
# 11. 'brightness'            → mild         (↓ 5–10%)
# 12. 'contrast'              → mild         (↓ 5–10%)
# 13. 'elastic_transform'     → mild/moderate(↓ 5–15%)
# 14. 'pixelate'              → moderate     (↓ 15–30%)
# 15. 'jpeg_compression'      → minimal      (↓ 0–5%)
# 16. 'rotate'                → severe       (↓ 40–60%)
# 17. 'shear'                 → moderate     (↓ 10–25%)
# 18. 'translate'             → moderate     (↓ 15–30%)
# 19. 'scale'                 → moderate     (↓ 15–30%)

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# === Load clean MNIST for training and validation ===
(x_train_full, y_train_full), (x_test_clean, y_test_clean) = tf.keras.datasets.mnist.load_data()
x_train_full = x_train_full.astype("float32") / 255.0
x_test_clean = x_test_clean.astype("float32") / 255.0

# Add channel dimension if needed
if x_train_full.shape[1:] == (28, 28):
    x_train_full = x_train_full[..., np.newaxis]
    x_test_clean = x_test_clean[..., np.newaxis]
    input_shape = (28, 28, 1)
else:
    raise ValueError("Unexpected image shape for MNIST: {}".format(x_train_full.shape))

y_train_full = y_train_full.flatten()
y_test_clean = y_test_clean.flatten()

# Split clean MNIST into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=seed
)
num_classes = len(np.unique(y_train_full))

# === Load impulse_noise corrupted MNIST for testing ===
builder = MNISTCorrupted(config="impulse_noise")
builder.download_and_prepare()
corrupted_ds = builder.as_dataset(split="test", as_supervised=True)
X_corrupted, y_corrupted = zip(*[(x, y) for x, y in tfds.as_numpy(corrupted_ds)])
X_corrupted = np.array(X_corrupted).astype("float32") / 255.0
y_corrupted = np.array(y_corrupted).flatten()
if X_corrupted.shape[1:] == (28, 28):
    X_corrupted = X_corrupted[..., np.newaxis]
elif X_corrupted.shape[1:] != (28, 28, 1):
    raise ValueError("Unexpected image shape for Corrupted MNIST: {}".format(X_corrupted.shape))

#%% Train DE-BNN on clean MNIST
ensemble = train_deep_ensemble(
    x_train, y_train, x_val, y_val, input_shape, num_classes, n_models=5, epochs=10
)

#%% Train Single-BNN on clean MNIST
single_bnn = ConvolutionalBNN(
    input_shape=input_shape,
    num_classes=num_classes,
    len_x_train=len(x_train),
    class_labels=None,
    seed=seed
)
single_bnn.compile()
single_bnn.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=64,
    verbose=1
)

#%% Evaluate both models on corrupted MNIST
print("\nEvaluating DE-BNN on Corrupted MNIST")
y_proba_de = ensemble_predict_proba(ensemble, X_corrupted)
results_de = evaluate_model(y_corrupted, y_proba_de)

print("\nEvaluating Single-BNN on Corrupted MNIST")
y_proba_single = single_bnn.predict_proba(X_corrupted)
results_single = evaluate_model(y_corrupted, y_proba_single)

#%% Print comparison
results_df = pd.DataFrame({
    "Single-BNN": results_single,
    "DE-BNN": results_de
})
print("\nComparison of Evaluation Metrics on Corrupted MNIST:")
print(results_df)

# 1. Get predicted probabilities from each ensemble model
probs_list = [model.predict_proba(X_corrupted) for model in ensemble]

# 2. Evaluate for each ensemble size (1 to 5)
results_per_size = []
for k in range(1, len(ensemble) + 1):
    probs_k = np.mean(probs_list[:k], axis=0)
    metrics = evaluate_model(y_corrupted, probs_k, num_classes=num_classes)
    metrics["Ensemble Size"] = k
    results_per_size.append(metrics)

df_ensemble_metrics = pd.DataFrame(results_per_size)
print(df_ensemble_metrics)

plots = plot_ensemble_metrics(df_ensemble_metrics, evaluate_model(y_corrupted, y_proba_single, num_classes=num_classes), mnist=True)
save_plots(plots, output_dir="plots/Saved_Plots/Plots_fair_comparison", file_suffix="Corrupted")
