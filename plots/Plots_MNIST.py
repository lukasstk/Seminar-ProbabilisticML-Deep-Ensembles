from plots.Plots_Helper import plot_ensemble_metrics, save_plots
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from Save_and_Load_Models import save_bnn_model
from Ensemble_helper import train_deep_ensemble, evaluate_model, ensemble_predict_proba


seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#%% Load MNIST and apply preprocessing
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train_full = x_train_full[..., np.newaxis]
x_test = x_test[..., np.newaxis]
input_shape = (28, 28, 1)

y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=seed)
num_classes = len(np.unique(y_train_full))

#%% Train DE-BNN
ensemble = train_deep_ensemble(x_train, y_train, x_val, y_val, input_shape, num_classes, n_models=5)

#%% Evaluate DE-BNN
print("\nEvaluating DE-BNN")
y_proba_de = ensemble_predict_proba(ensemble, x_test)
results_de = evaluate_model(y_test, y_proba_de)

#%% Print comparison
results_df = pd.DataFrame({
    "DE-BNN": results_de
})
print("\nComparison of Evaluation Metrics:")
print(results_df)

#%% Evaluation per ensemble size
probs_list = [model.predict_proba(x_test) for model in ensemble]

results_per_size = []
for k in range(1, len(ensemble) + 1):
    probs_k = np.mean(probs_list[:k], axis=0)
    metrics = evaluate_model(y_test, probs_k, num_classes=10)
    metrics["Ensemble Size"] = k
    results_per_size.append(metrics)

df_ensemble_metrics = pd.DataFrame(results_per_size)
print(df_ensemble_metrics)

plots = plot_ensemble_metrics(df_ensemble_metrics, df_ensemble_metrics.iloc[0, :], mnist=True)

save_plots(plots)

#%% Save models
for i, model in enumerate(ensemble):
    save_bnn_model(model, f"Ensemble_Member_{i+1}_MNIST")
