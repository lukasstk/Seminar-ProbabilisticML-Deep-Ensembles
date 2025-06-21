from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import pyplot as plt

#%% Angepasste Klasse für großes BNN
class BigConvolutionalBNN(ConvolutionalBNN):
    def _build_model(self):
        kl = lambda q, p, _: tfp.distributions.kl_divergence(q, p) * self.kl_weight
        model = tfk.Sequential([
            tfpl.Convolution2DFlipout(filters=64, kernel_size=(3, 3), activation='relu',
                                      input_shape=self.input_shape, kernel_divergence_fn=kl),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),
            tfpl.Convolution2DFlipout(filters=128, kernel_size=(3, 3), activation='relu', kernel_divergence_fn=kl),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),
            tfpl.Convolution2DFlipout(filters=256, kernel_size=(3, 3), activation='relu', kernel_divergence_fn=kl),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),
            tfk.layers.Flatten(),
            tfpl.DenseFlipout(units=512, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.3),
            tfpl.DenseFlipout(units=256, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.2),
            tfpl.DenseFlipout(self.num_classes, activation=None, kernel_divergence_fn=kl)
        ])
        return model

#%% Deep Ensemble Training
def train_deep_ensemble(X_train, y_train, X_val, y_val, input_shape, num_classes, n_models, class_labels=None):
    ensemble_models = []
    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}")
        model = ConvolutionalBNN(input_shape, num_classes, len(X_train), class_labels=class_labels)
        model.compile()
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=64, verbose=1)
        ensemble_models.append(model)
    return ensemble_models

def ensemble_predict_proba(ensemble, X):
    all_probs = [model.predict_proba(X) for model in ensemble]
    return np.mean(np.stack(all_probs, axis=0), axis=0)

def ensemble_predict_classes(ensemble, X):
    probs = ensemble_predict_proba(ensemble, X)
    return np.argmax(probs, axis=1)

#%% Evaluation Utilities
def brier_score(y_true, y_proba, num_classes):
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_proba - y_true_one_hot), axis=1)).numpy()

def predictive_entropy(y_proba):
    dist = tfp.distributions.Categorical(probs=y_proba)
    return tf.reduce_mean(dist.entropy()).numpy()

def expected_calibration_error(y_true, y_proba, n_bins=10):
    confidences = tf.reduce_max(y_proba, axis=1)
    predictions = tf.argmax(y_proba, axis=1, output_type=tf.int32)
    accuracies = tf.cast(tf.equal(predictions, y_true), tf.float32)

    bin_boundaries = tf.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = tf.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = tf.reduce_mean(tf.cast(in_bin, tf.float32))
        if prop_in_bin > 0:
            accuracy_in_bin = tf.reduce_mean(tf.boolean_mask(accuracies, in_bin))
            avg_confidence_in_bin = tf.reduce_mean(tf.boolean_mask(confidences, in_bin))
            ece += tf.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.numpy()

def evaluate_model(y_true, y_proba, num_classes=10):
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_true, y_pred)
    nll = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_proba).numpy()
    brier = brier_score(y_true, y_proba, num_classes)
    entropy_val = predictive_entropy(y_proba)
    ece_val = expected_calibration_error(y_true, y_proba)
    return {
        "Accuracy": acc,
        "NLL": nll,
        "Brier Score": brier,
        "Predictive Entropy": entropy_val,
        "ECE": ece_val
    }

#%% Load and prepare MNIST
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train_full = x_train_full[..., np.newaxis]
x_test = x_test[..., np.newaxis]
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.1, random_state=42)
input_shape = (28, 28, 1)
num_classes = 10

#%% Train MFVI-BNN
print("\nTraining Large BNN-VI:")
bnn_big = BigConvolutionalBNN(input_shape, num_classes, len(x_train))
bnn_big.compile()
bnn_big.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=64)

#%% Train DE-BNN
ensemble = train_deep_ensemble(x_train, y_train, x_val, y_val, input_shape, num_classes, n_models=10)

#%% Evaluate both models
print("\nEvaluating Large BNN-VI")
y_proba_big = bnn_big.predict_proba(x_test)
results_big = evaluate_model(y_test, y_proba_big)

print("\nEvaluating DE-BNN")
y_proba_de = ensemble_predict_proba(ensemble, x_test)
results_de = evaluate_model(y_test, y_proba_de)

#%% Print comparison
results_df = pd.DataFrame({
    "MFVI-BNN": results_big,
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

def plot_ensemble_metrics(df, mfvi_reference):
    """
    Plots evaluation metrics vs. ensemble size from a DataFrame and compares them to a fixed MFVI-BNN baseline.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns:
        - "Ensemble Size"
        - "Accuracy", "NLL", "Brier Score", "Predictive Entropy", "ECE"

    mfvi_reference : dict
        Reference metric values from a single MFVI-BNN model. Keys must match column names in df.
    """
    ensemble_sizes = df["Ensemble Size"].tolist()

    # Automatically detect metric columns (exclude "Ensemble Size")
    metric_columns = [col for col in df.columns if col != "Ensemble Size"]

    for metric_name in metric_columns:
        values = df[metric_name].tolist()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ensemble_sizes, values, marker='o', color='blue', label='DE-BNN', linewidth=2.5, markersize=8)
        ax.axhline(y=mfvi_reference[metric_name], color='red', linestyle='--', label='MFVI-BNN', linewidth=2.5)
        ax.set_xlabel("Ensemble Size")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs. Ensemble Size")
        ax.set_xticks(ensemble_sizes)
        ax.legend(fontsize=20)
        ax.grid(True)
        fig.tight_layout()
        apply_custom_theme(ax)

        # Save to plots/Saved_Plots
        filename = f"plots/Saved_Plots/{metric_name.replace(' ', '_').lower()}.pdf"
        fig.savefig(filename, bbox_inches="tight", dpi=300)

        plt.show()

plot_ensemble_metrics(df_ensemble_metrics, results_big)