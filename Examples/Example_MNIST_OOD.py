from matplotlib import pyplot as plt
from Model_Code.Save_and_Load_Models import load_bnn_model
from Model_Code.Ensemble_helper import evaluate_model, ensemble_predict_proba, train_deep_ensemble
from plots.Custom_plot_style import apply_custom_theme
from sklearn.model_selection import train_test_split
import tensorflow             as tf
import numpy                  as np
import pandas                 as pd
import random, os
import matplotlib.ticker as mtick

from plots.Plots_Helper import save_plots

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# --- In-Distribution (clean MNIST) -------------------------------------------
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_full      = x_train_full.astype("float32")      / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train_full      = x_train_full[..., np.newaxis]                       # (N,28,28,1)
x_test  = x_test[..., np.newaxis]


x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=seed)

# --- OOD-Data (Fashion-MNIST) -----------------------------------------------
(x_ood_train, y_ood_train), (x_ood_test, y_ood_test) = tf.keras.datasets.fashion_mnist.load_data()

x_ood_test = (x_ood_test.astype("float32") / 255.)[..., np.newaxis]

input_shape = (28, 28, 1)
num_classes = 10

# -- Deep Ensemble ------------------------------------------------------------
ensemble = train_deep_ensemble(
    x_train, y_train,
    x_val,   y_val,
    input_shape, num_classes,
    n_models=5,
)

"""ensemble = [
    load_bnn_model(f"Ensemble_Member_{i+1}_MNIST", len_x_train=len(x_train_full))
    for i in range(5)
]"""

y_proba_de = ensemble_predict_proba(ensemble, x_test)
y_proba_de_ood = ensemble_predict_proba(ensemble, x_ood_test)

results_de = evaluate_model(y_true = y_test, y_proba = y_proba_de, num_classes = num_classes)
results_de_ood = evaluate_model(y_true = y_test, y_proba = y_proba_de_ood, num_classes = num_classes)

probs_list_id  = [m.predict_proba(x_test) for m in ensemble]   # In-Distribution
probs_list_ood = [m.predict_proba(x_ood_test)  for m in ensemble]   # Out-of-Distribution

metrics_id  = []
metrics_ood = []

for k in range(1, len(ensemble) + 1):
    probs_k_id  = np.mean(probs_list_id[:k],  axis=0)
    probs_k_ood = np.mean(probs_list_ood[:k], axis=0)

    m_id  = evaluate_model(y_test, probs_k_id,  num_classes)   # In-D
    m_ood = evaluate_model(y_ood_test,  probs_k_ood, num_classes)   # OOD

    m_id["Ensemble Size"]  = k
    m_ood["Ensemble Size"] = k

    metrics_id.append(m_id)
    metrics_ood.append(m_ood)

df_ensemble_metrics_id  = pd.DataFrame(metrics_id)
df_ensemble_metrics_ood = pd.DataFrame(metrics_ood)

print("\nID-Kurve:\n",  df_ensemble_metrics_id)
print("\nOOD-Kurve:\n", df_ensemble_metrics_ood)

ensemble_sizes = df_ensemble_metrics_id["Ensemble Size"].tolist()          # [1,2,3,4,5]
entropy_id     = df_ensemble_metrics_id["Predictive Entropy"].tolist()     # In-D
entropy_ood    = df_ensemble_metrics_ood["Predictive Entropy"].tolist()    # OOD

# === Plot ====================================================================
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ensemble_sizes, entropy_id,  marker="o", linewidth=2.5,
        markersize=8, color="mediumorchid",  label="In-Distribution")
ax.plot(ensemble_sizes, entropy_ood, marker="s", linewidth=2.5,
        markersize=8, color="gold",   label="Out-of-Distribution")

ax.set_xlabel("Ensemble Size")
ax.set_ylabel("Predictive Entropy")
ax.set_title("Predictive Entropy: ID vs. OOD")
ax.set_xticks(ensemble_sizes)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
ax.legend(fontsize=14)
ax.grid(True)
fig.tight_layout()
apply_custom_theme(ax)
plt.show()

# Create a list in the expected format: [(fig, metric_name, suffix)]
plots = [
    (fig, "predictive_entropy_entropy_ID_vs_OOD", ".png")  # suffix must start with dot if you want .png
]

# Save the plot using your function
save_plots(plots, output_dir="plots/Saved_Plots")



