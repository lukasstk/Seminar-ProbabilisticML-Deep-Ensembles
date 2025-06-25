from plots.Plots_Helper import plot_ensemble_metrics
from imports import *
from sklearn.model_selection import train_test_split
import pandas as pd
from Save_and_Load_Models import save_bnn_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from Ensemble_helper import train_deep_ensemble, BigConvolutionalBNN, evaluate_model, ensemble_predict_proba
import matplotlib.pyplot as plt

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#%% Load MNIST or CIFAR-10 and apply preprocessing
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # Uncomment if using MNIST

# Normalize pixel values
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Detect dataset type based on image shape and adjust channels
if x_train_full.shape[1:] == (28, 28):  # MNIST
    x_train_full = x_train_full[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    input_shape = (28, 28, 1)
elif x_train_full.shape[1:] == (32, 32, 3):  # CIFAR-10
    input_shape = (32, 32, 3)
else:
    raise ValueError("Unrecognized dataset shape: {}".format(x_train_full.shape[1:]))

# Flatten labels
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Split into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=seed)
num_classes = len(np.unique(y_train_full))

"""ensemble = []"""
"""ensemble.append(load_bnn_model("Ensemble_Member_1", len_x_train=len(x_train_full)))
ensemble.append(load_bnn_model("Ensemble_Member_2", len_x_train=len(x_train_full)))
ensemble.append(load_bnn_model("Ensemble_Member_3", len_x_train=len(x_train_full)))
ensemble.append(load_bnn_model("Ensemble_Member_4", len_x_train=len(x_train_full)))
ensemble.append(load_bnn_model("Ensemble_Member_5", len_x_train=len(x_train_full)))"""

"""ensemble = [
    load_bnn_model(f"Ensemble_Member_{i+1}_CIFAR10", len_x_train=len(x_train_full))
    for i in range(5)
]"""


#%% Train DE-BNN
ensemble = train_deep_ensemble(x_train, y_train, x_val, y_val, input_shape, num_classes, n_models=5)

"""# Modell initialisieren
CIFAR10_BNN = BigConvolutionalBNN(input_shape, num_classes, len_x_train=len(x_train_full))

# Kompilieren mit angepasster Lernrate
CIFAR10_BNN.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# EarlyStopping-Callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
    #, min_delta=0.001 # Optional: minimale Verbesserung von 0.1 Prozentpunkten
)

# Training starten mit Callback
CIFAR10_BNN.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

y_proba_cifar = CIFAR10_BNN.predict_proba_mc(x_test, mc_samples=30)
results_cifar_bnn = evaluate_model(y_test, y_proba_cifar)
print(results_cifar_bnn)"""

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

plot_ensemble_metrics(df_ensemble_metrics, df_ensemble_metrics.iloc[0,:], mnist=False)

save_bnn_model(ensemble[0], "Ensemble_Member_1_CIFAR10")
save_bnn_model(ensemble[1], "Ensemble_Member_2_CIFAR10")
save_bnn_model(ensemble[2], "Ensemble_Member_3_CIFAR10")
save_bnn_model(ensemble[3], "Ensemble_Member_4_CIFAR10")
save_bnn_model(ensemble[4], "Ensemble_Member_5_CIFAR10")