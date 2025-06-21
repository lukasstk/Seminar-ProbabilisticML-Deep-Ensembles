from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN
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


#%% Deep Ensemble wie gehabt
def train_deep_ensemble(X_train, y_train, X_val, y_val, input_shape, num_classes, n_models, class_labels=None):
    ensemble_models = []
    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}")
        model = ConvolutionalBNN(input_shape, num_classes, len(X_train), class_labels=class_labels)
        model.compile()
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=64, verbose=1)
        ensemble_models.append(model)
    return ensemble_models

#%% Ensemble-Vorhersagen
def ensemble_predict_logits(ensemble, X):
    all_logits = [model.predict_logits(X) for model in ensemble]
    return np.stack(all_logits, axis=0)

def ensemble_predict_proba(ensemble, X):
    logits = ensemble_predict_logits(ensemble, X)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    return np.mean(probs, axis=0)

def ensemble_predict_classes(ensemble, X):
    probs = ensemble_predict_proba(ensemble, X)
    return np.argmax(probs, axis=1)

#%% MNIST vorbereiten
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

input_shape = (28, 28, 1)
num_classes = 10

#%% --- Trainiere großes BNN-VI (gleiche Rechenkapazität wie DE-BNN) ---
print("\nTraining Large BNN-VI:")
bnn_big = BigConvolutionalBNN(input_shape, num_classes, len(X_train))
bnn_big.compile()
bnn_big.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=64)

#%% --- Trainiere DE-BNN (Ensemble aus 5 kleinen BNNs) ---
ensemble = train_deep_ensemble(X_train, y_train, X_val, y_val, input_shape, num_classes, n_models=5)

#%% --- Vergleich der Accuracy ---
acc_big_bnn = bnn_big.evaluate_accuracy(X_test, y_test)
acc_ensemble = accuracy_score(y_test, ensemble_predict_classes(ensemble, X_test))

print(f"\nLarge BNN-VI Accuracy: {acc_big_bnn:.4f}")
print(f"DE-BNN Accuracy:       {acc_ensemble:.4f}")

# --- Evaluation Utilities ---



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

# --- Evaluation wrapper ---
def evaluate_model(y_true, y_proba, num_classes=10):
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_true, y_pred)
    nll = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_proba).numpy()
    brier = brier_score(y_true, y_proba, num_classes)
    entropy_val = predictive_entropy(y_proba)
    ece_val = expected_calibration_error(y_true, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"NLL: {nll:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Predictive Entropy: {entropy_val:.4f}")
    print(f"Expected Calibration Error (ECE): {ece_val:.4f}")

    return {
        "Accuracy": acc,
        "NLL": nll,
        "Brier Score": brier,
        "Predictive Entropy": entropy_val,
        "ECE": ece_val
    }

#%% ---- Run Evaluation ----


print("\nEvaluating Large BNN-VI")
y_proba_big = tf.nn.softmax(bnn_big.predict_logits(X_test), axis=-1).numpy()
results_big = evaluate_model(y_test, y_proba_big)

print("\nEvaluating DE-BNN")
y_proba_de = ensemble_predict_proba(ensemble, X_test)
results_de = evaluate_model(y_test, y_proba_de)
