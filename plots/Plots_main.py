from Model_Code.ConvolutionalBNN_Model import ConvolutionalBNN
from plots.Custom_plot_style import apply_custom_theme
from plots.Plots_Helper import plot_ensemble_metrics, save_plots
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from Model_Code.Save_and_Load_Models import save_bnn_model, load_bnn_model
from Model_Code.Ensemble_helper import train_deep_ensemble, evaluate_model, ensemble_predict_proba
import matplotlib.ticker as mtick

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#%% Load MNIST or CIFAR-10 and apply preprocessing
#(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # Uncomment if using MNIST

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

#Load already existing models, very similar but not identical results
ensemble = []

"""ensemble = [
    load_bnn_model(f"Ensemble_Member_{i+1}_CIFAR10", len_x_train=len(x_train_full))
    for i in range(5)
]"""
# or for MNIST
ensemble = [
    load_bnn_model(f"Ensemble_Member_{i+1}_MNIST", len_x_train=len(x_train_full))
    for i in range(5)
]

#%% Train DE-BNN

# ensemble = train_deep_ensemble(x_train, y_train, x_val, y_val, input_shape, num_classes, epochs = 10, n_models=5)

# 1. Instantiate your model
single_bnn = ConvolutionalBNN(
    input_shape=input_shape,
    num_classes=num_classes,
    len_x_train=len(x_train),     # or len(x_train_full) if you want to use the full train set length for KL
    class_labels=None,            # or [0, 1, ..., 9] if you want explicit labels
    seed=seed
)

# 2. Compile your model
single_bnn.compile()

# 3. Train the model
single_bnn.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=50,         # or any desired number of epochs
    batch_size=64,    # or any suitable batch size
    verbose=1
)

#%% Evaluate both models
print("\nEvaluating DE-BNN")
y_proba_de = ensemble_predict_proba(ensemble, x_test)
results_de = evaluate_model(y_test, y_proba_de)

print("\nEvaluating Single-BNN")
y_proba_single = single_bnn.predict_proba(x_test)
results_single = evaluate_model(y_test, y_proba_single)

#%% Print comparison
results_df = pd.DataFrame({
    "Single-BNN": results_single,
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

plots = plot_ensemble_metrics(df_ensemble_metrics, evaluate_model(y_test, y_proba_single, num_classes=10) , mnist=True)

"""save_plots(plots, output_dir="plots/Saved_Plots")"""

#Train/Val/Test Accuracy plot
num_epochs = 50
batch_size = 64

train_accs = []
val_accs = []
test_accs = []

for epoch in range(num_epochs):
    # Accuracy on train, val, test
    train_acc = single_bnn.evaluate_accuracy(x_train, y_train)
    val_acc = single_bnn.evaluate_accuracy(x_val, y_val)
    test_acc = single_bnn.evaluate_accuracy(x_test, y_test)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch + 1:02d}: Train={train_acc:.4f}  Val={val_acc:.4f}  Test={test_acc:.4f}")

epochs = np.arange(1, num_epochs + 1)
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(epochs, train_accs, label='Train Accuracy', color = "dodgerblue")
ax.plot(epochs, val_accs, label='Validation Accuracy', color = "darkorange")
ax.plot(epochs, test_accs, label='Test Accuracy', color = "darkslategray")

ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Learning Curve (Train/Val/Test)', fontweight='bold')

ax.set_xlim(0, 50)
ax.set_ylim(0.75, 0.96)  # Optional: set this tighter to your range, or use 0,1 for full percent scale

# Format y-axis as percent
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
ax.set_xticks(np.arange(0, 51, 5))

# Bigger legend
ax.legend(fontsize=16, loc='lower right')

ax.grid(True)
fig.tight_layout()

apply_custom_theme(ax)
plt.show()

all_acc_single = [
    (fig, "all_acc_single_bnn", ".png")   # Dateiname: learning_curve_single_bnn.png
]
save_plots(all_acc_single, output_dir="plots/Saved_Plots")

save_bnn_model(single_bnn, "Single_Model_MNIST")

"""save_bnn_model(ensemble[0], "Ensemble_Member_1_MNIST")
save_bnn_model(ensemble[1], "Ensemble_Member_2_MNIST")
save_bnn_model(ensemble[2], "Ensemble_Member_3_MNIST")
save_bnn_model(ensemble[3], "Ensemble_Member_4_MNIST")
save_bnn_model(ensemble[4], "Ensemble_Member_5_MNIST")"""