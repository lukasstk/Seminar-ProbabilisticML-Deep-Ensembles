from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN
from sklearn.metrics import accuracy_score
from netcal.metrics import ECE
from sklearn.metrics import brier_score_loss
from tensorflow.keras.callbacks import EarlyStopping

#%% Angepasste Klasse für CIFAR-10 BNN
class BigConvolutionalBNN(ConvolutionalBNN):
    def _build_model(self):
        kl = lambda q, p, _: tfp.distributions.kl_divergence(q, p) * self.kl_weight

        model = tfk.Sequential([
            # Block 1
            tfpl.Convolution2DFlipout(64, kernel_size=(3, 3), padding='same', activation='relu',
                                      input_shape=self.input_shape, kernel_divergence_fn=kl),
            tfk.layers.BatchNormalization(),
            tfpl.Convolution2DFlipout(64, kernel_size=(3, 3), padding='same', activation='relu',
                                      kernel_divergence_fn=kl),
            tfk.layers.BatchNormalization(),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),

            # Block 2
            tfpl.Convolution2DFlipout(128, kernel_size=(3, 3), padding='same', activation='relu',
                                      kernel_divergence_fn=kl),
            tfk.layers.BatchNormalization(),
            tfpl.Convolution2DFlipout(128, kernel_size=(3, 3), padding='same', activation='relu',
                                      kernel_divergence_fn=kl),
            tfk.layers.BatchNormalization(),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),

            # Block 3
            tfpl.Convolution2DFlipout(256, kernel_size=(3, 3), padding='same', activation='relu',
                                      kernel_divergence_fn=kl),
            tfk.layers.BatchNormalization(),
            tfpl.Convolution2DFlipout(256, kernel_size=(3, 3), padding='same', activation='relu',
                                      kernel_divergence_fn=kl),
            tfk.layers.BatchNormalization(),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),

            # Dense
            tfk.layers.Flatten(),
            tfpl.DenseFlipout(512, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.4),
            tfpl.DenseFlipout(256, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.3),
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
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, verbose=1)
        ensemble_models.append(model)
    return ensemble_models

def ensemble_predict_proba(ensemble, X):
    all_probs = [model.predict_proba(X) for model in ensemble]
    return np.mean(np.stack(all_probs, axis=0), axis=0)

def ensemble_predict_classes(ensemble, X):
    probs = ensemble_predict_proba(ensemble, X)
    return np.argmax(probs, axis=1)

def predictive_entropy(y_proba):
    dist = tfp.distributions.Categorical(probs=y_proba)
    return tf.reduce_mean(dist.entropy()).numpy()

def evaluate_model(y_true, y_proba, num_classes=10):
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_true, y_pred)
    nll = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_proba).numpy()
    brier = brier_score_loss(y_true, y_proba, scale_by_half=False)
    entropy_val = predictive_entropy(y_proba)

    ece = ECE(bins=num_classes)
    ece_val = ece.measure(y_proba, y_true)
    return {
        "Accuracy": acc,
        "NLL": nll,
        "Brier Score": brier,
        "Predictive Entropy": entropy_val,
        "ECE": ece_val
    }