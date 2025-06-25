import random
from tensorflow_datasets.image_classification import MNISTCorrupted
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
from Ensemble_helper import train_deep_ensemble, evaluate_model, ensemble_predict_proba
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