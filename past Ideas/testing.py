import tensorflow as tf
import tf_keras as tfk
import tensorflow_probability as tfp
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

tfpl = tfp.layers

class ConvolutionalBNN:
    def __init__(self, input_shape, kl_weight, class_labels=None):
        self.input_shape = input_shape
        self.kl_weight = kl_weight

        self.model = None
        self._compile_args = {}
        self._handle_int_label_aliases = False
        self.index_to_label = None
        self.label_to_index = None
        self.class_labels = class_labels
        self.num_classes = len(class_labels) if class_labels is not None else None

    def _build_model(self):
        kl = lambda q, p, _: tfp.distributions.kl_divergence(q, p) * self.kl_weight

        model = tfk.Sequential([
            tfpl.Convolution2DFlipout(32, (3, 3), activation='relu',
                                      input_shape=self.input_shape, kernel_divergence_fn=kl),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),

            tfpl.Convolution2DFlipout(64, (3, 3), activation='relu', kernel_divergence_fn=kl),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),

            tfpl.Convolution2DFlipout(128, (3, 3), activation='relu', kernel_divergence_fn=kl),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),

            tfk.layers.Flatten(),
            tfpl.DenseFlipout(256, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.3),
            tfpl.DenseFlipout(128, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.2),
            tfpl.DenseFlipout(self.num_classes, activation=None, kernel_divergence_fn=kl)
        ])
        return model

    def compile(self, optimizer="adam", loss=None, metrics=None, from_logits=True):
        self._compile_args = {
            "optimizer": optimizer,
            "loss": loss or tfk.losses.SparseCategoricalCrossentropy(from_logits=from_logits),
            "metrics": metrics or ["accuracy"]
        }
        if self.model is not None:
            self.model.compile(**self._compile_args)

    def fit(self, X_train, y_train, shuffle_data=True, random_state=None, validation_data=None, **kwargs):
        # Detect class labels and set mappings if not already provided
        if self.class_labels is None:
            unique_labels = sorted(set(y_train))
            if all(isinstance(label, str) for label in unique_labels):
                self.class_labels = unique_labels
            else:
                self.class_labels = list(range(len(unique_labels)))

        self.num_classes = len(self.class_labels)
        self.index_to_label = {i: label for i, label in enumerate(self.class_labels)}
        self.label_to_index = {label: i for i, label in self.index_to_label.items()}
        self._handle_int_label_aliases = all(isinstance(label, str) for label in self.class_labels)

        # Build model and compile after class labels are known
        if self.model is None:
            self.model = self._build_model()
            self.model.compile(**self._compile_args)

        # Convert y_train to integer indices
        if isinstance(y_train[0], str):
            y_train = np.array([self.label_to_index[label] for label in y_train])
        elif self._handle_int_label_aliases:
            y_train = np.array([self.label_to_index[self.index_to_label[label]] for label in y_train])

        # Convert validation labels
        if validation_data is not None:
            X_val, y_val = validation_data
            if isinstance(y_val[0], str):
                y_val = np.array([self.label_to_index[label] for label in y_val])
            elif self._handle_int_label_aliases:
                y_val = np.array([self.label_to_index[self.index_to_label[label]] for label in y_val])
            validation_data = (X_val, y_val)

        # Shuffle if needed
        if shuffle_data:
            if random_state is None:
                random_state = random.randint(0, 10_000_000)
            X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

        self.model.fit(X_train, y_train, validation_data=validation_data, **kwargs)

    def predict_logits(self, X_test):
        return self.model(X_test, training=False).numpy()

    def predict_classes(self, X_test):
        logits = self.predict_logits(X_test)
        class_indices = np.argmax(logits, axis=1)
        return [self.index_to_label[int(i)] for i in class_indices]

    def evaluate_accuracy(self, X_test, y_test):
        y_pred = self.predict_classes(X_test)
        if isinstance(y_test[0], str):
            return accuracy_score(y_test, y_pred)
        else:
            y_test_str = [self.index_to_label[int(i)] for i in y_test]
            return accuracy_score(y_test_str, y_pred)
# --- Test it on MNIST with animal labels ---

# Load MNIST
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_full = x_train_full[..., tf.newaxis] / 255.0
x_test = x_test[..., tf.newaxis] / 255.0

# Split
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Use animal labels instead of digits
animal_labels = ["cat", "dog", "owl", "lion", "tiger", "mouse", "bear", "fox", "cow", "duck"]
y_train_str = [animal_labels[i] for i in y_train]
y_val_str = [animal_labels[i] for i in y_val]
y_test_str = [animal_labels[i] for i in y_test]

# Create and train model WITHOUT passing class_labels
bnn = ConvolutionalBNN(input_shape=(28, 28, 1), kl_weight=1.0 / len(x_train))
bnn.compile()
bnn.fit(x_train, y_train_str, validation_data=(x_val, y_val_str), epochs=2, batch_size=128)

# Predict
print("Predicted:", bnn.predict_classes(x_test[:10]))
print("Actual:", y_test_str[:10])
print("Test accuracy:", bnn.evaluate_accuracy(x_test, y_test_str))