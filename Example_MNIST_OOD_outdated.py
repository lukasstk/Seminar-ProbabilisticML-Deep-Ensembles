from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from DE_BNN_vs_single_BNN import *

# === 1. Trainiere neues Single-BNN und Ensemble auf klassischem MNIST ===
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

input_shape = (28, 28, 1)
num_classes = 10

# Train single BNN
bnn = ConvolutionalBNN(input_shape, num_classes, len(X_train))
bnn.compile()
bnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=128)

# Train DE-BNNs
ensemble = train_deep_ensemble(X_train, y_train, X_val, y_val, input_shape, num_classes, n_models=5)

# === 2. Lade OOD-Daten (Fashion-MNIST) ===
(X_ood, y_ood), _ = tf.keras.datasets.fashion_mnist.load_data()
X_ood = X_ood.astype("float32") / 255.
X_ood = X_ood[..., np.newaxis]

# === 3. Berechne Entropie der Ensemble-Vorhersagen auf OOD-Daten ===
probs_ood = ensemble_predict_proba(ensemble, X_ood)
entropy_ood = entropy(probs_ood.T)  # shape: (n_samples,)

# === 4. Berechne Entropie der Ensemble-Vorhersagen auf In-Distribution (MNIST) ===
probs_ind = ensemble_predict_proba(ensemble, X_test)
entropy_ind = entropy(probs_ind.T)

# === 5. Ausgabe statistischer Kennzahlen ===
print("\n🔍 Predictive Entropy (Mean ± Std)")
print(f"→ In-Distribution (MNIST):     {entropy_ind.mean():.4f} ± {entropy_ind.std():.4f}")
print(f"→ Out-of-Distribution (Fashion-MNIST): {entropy_ood.mean():.4f} ± {entropy_ood.std():.4f}")
