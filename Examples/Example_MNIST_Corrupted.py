import random
from tensorflow_datasets.image_classification import MNISTCorrupted
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
from Model_Code.Ensemble_helper import train_deep_ensemble, evaluate_model, ensemble_predict_proba
from plots.Plots_Helper import plot_ensemble_metrics, save_plots
import pandas as pd
import tensorflow as tf

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# === Load impulse_noise corrupted MNIST ===
builder = MNISTCorrupted(config="impulse_noise")
builder.download_and_prepare()

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


ds = builder.as_dataset(split="test", as_supervised=True)
X_all, y_all = zip(*[(x, y) for x, y in tfds.as_numpy(ds)])
X_all = np.array(X_all) / 255.0
y_all = np.array(y_all)

# === Split into train/val/test (e.g., 60/20/20 split) ===
x_train_full, x_test, y_train_full, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=seed)

input_shape  = (28, 28, 1)
num_classes  = 10

#%% Train DE-BNN
ensemble = train_deep_ensemble(x_train, y_train, x_val, y_val, input_shape, num_classes, n_models=5)

#%% Evaluate both models
print("\nEvaluating DE-BNN")
y_proba_de = ensemble_predict_proba(ensemble, x_test)
results_de = evaluate_model(y_test, y_proba_de)

#%% Print comparison
results_df = pd.DataFrame({
    """"Single-BNN": results_single,"""
    "DE-BNN": results_de
})
print("\nComparison of Evaluation Metrics:")
print(results_df)

# 1. Get predicted probabilities from each ensemble model
probs_list = [model.predict_proba(x_test) for model in ensemble]

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

plots = plot_ensemble_metrics(df_ensemble_metrics, df_ensemble_metrics.iloc[0,:], mnist=True)

save_plots(plots, file_suffix="Corrupted")