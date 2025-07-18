import random
import numpy as np
import tensorflow as tf
import tf_keras as tfk
import tensorflow_probability as tfp
tfpl = tfp.layers
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


class ConvolutionalBNN:
    def __init__(self, input_shape, num_classes, len_x_train, class_labels=None, seed=42):
        """
        Args:
            input_shape: tuple, shape of input images (e.g., (28, 28, 1))
            kl_weight: float, scaling factor for KL divergence regularization
            num_classes: int, number of output classes (e.g., 10)
            class_labels: list, optional list mapping indices to class labels (e.g., [1,2,...,11] or ["cat", "dog", ...])
            len_x_train: int, length of x_train (trainigset)
        """

        self.label_to_index = None
        self.index_to_label = None # is defined in .fit() after class_labels is defined for sure

        self.kl_weight = 1.0 / len_x_train
        self.input_shape = input_shape
        self.class_labels = class_labels
        self.num_classes = num_classes
        self.model = self._build_model()

        # Seed for deterministic Sampling
        self.seed = seed
        self.seed_stream = tfp.util.SeedStream(seed=seed, salt="flipout")

    def _build_model(self):
        """
        Build a convolutional BNN using Flipout layers and KL divergence regularization.
        """

        data_augmentation = tfk.Sequential([
            tfk.layers.RandomRotation(0.1),
            tfk.layers.RandomTranslation(0.1, 0.1),
        ])

        kl = lambda q, p, _: tfp.distributions.kl_divergence(q, p) * self.kl_weight

        model = tfk.Sequential([

            data_augmentation,

            tfpl.Convolution2DFlipout(filters = 32, kernel_size = (3, 3), activation='relu',
                                      input_shape=self.input_shape, kernel_divergence_fn=kl),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),

            tfpl.Convolution2DFlipout(filters = 64, kernel_size = (3, 3), activation='relu', kernel_divergence_fn=kl),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),

            tfpl.Convolution2DFlipout(filters = 128, kernel_size = (3, 3), activation='relu', kernel_divergence_fn=kl),
            tfk.layers.MaxPooling2D(pool_size=(2, 2)),

            tfk.layers.Flatten(),
            tfpl.DenseFlipout(units = 256, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.3),
            tfpl.DenseFlipout(units = 128, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.2),
            tfpl.DenseFlipout(self.num_classes, activation=None, kernel_divergence_fn=kl)
        ])
        return model

    def compile(self, optimizer="adam", loss=None, metrics=None, from_logits=True):
        """
        Compile the model with customizable optimizer, loss, and metrics.

        Parameters:
        ----------
        optimizer : str or tf.keras.optimizers.Optimizer (default: 'adam')
            Which optimizer to use:
            - 'adam' (default): Good general-purpose optimizer; works well in most cases.
            - 'sgd' : Use for simpler or linearly separable problems; slower convergence.
            - 'rmsprop' : Often used in RNNs or problems with non-stationary objectives.

        loss : str or tf.keras.losses.Loss (default: 'sparse_categorical_crossentropy')
            Which loss function to use:
            - 'sparse_categorical_crossentropy' (default): Use when labels are integers (e.g., 0–9).
            - 'categorical_crossentropy': Use when labels are one-hot encoded vectors.

        metrics : list of str or tf.keras.metrics.Metric (default: ['accuracy'])
            Which metrics to track during training:
            - 'accuracy': Most common metric for classification.
            - 'precision': Use if false positives are costly.
            - 'recall': Use if false negatives are costly.

        from_logits : bool (default: True)
            Set to True if your model outputs logits (raw scores).
            Set to False if your model uses softmax in the last layer (outputs probabilities).
        """

        if optimizer is None:
            optimizer = "adam"

        if loss is None:
            loss = tfk.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
        if metrics is None:
            metrics = ["accuracy"]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, X_train, y_train, shuffle_data=True, random_state=None, validation_data=None, verbose=1, **kwargs):
        """
        Train the model.
        Accepts additional arguments like epochs=..., batch_size=..., etc.
        """

        # Detect class labels and set mappings if not already provided
        if self.class_labels is None:
            self.class_labels = sorted(set(y_train))

        self.index_to_label = {i: label for i, label in enumerate(self.class_labels)} # index -> label
        self.label_to_index = {label: i for i, label in self.index_to_label.items()} # label -> index

        # Convert y_train to integer indices
        if isinstance(y_train[0], str):
            y_train = np.array([self.label_to_index[label] for label in y_train])

        # Convert validation labels
        if validation_data is not None:
            X_val, y_val = validation_data
            if isinstance(y_val[0], str):
                y_val = np.array([self.label_to_index[label] for label in y_val])
            validation_data = (X_val, y_val)

        # Shuffle if needed
        if shuffle_data:
            if random_state is None:
                random_state = random.randint(0, 10_000_000)
            X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

        history = self.model.fit(X_train, y_train, validation_data=validation_data, verbose=verbose, **kwargs)
        return history

    def predict_logits(self, X_test):
        """
        Returns raw logits (before softmax) for X_test.
        Shape: (num_samples, num_classes)
        """
        return self.model(X_test, training=False).numpy()

    def predict_proba(self, X_test):
        """
        Returns predicted class probabilities (after softmax).
        Shape: (num_samples, num_classes)
        """
        logits = self.predict_logits(X_test)
        return tf.nn.softmax(logits, axis=-1).numpy()

    def predict_proba_mc(self, X_test, mc_samples=30, use_seed=True):
        """
        Predicts class probabilities by averaging over multiple stochastic forward passes.

        Args:
            X_test : np.ndarray
                Test data
            mc_samples : int
                Number of MC samples
            use_seed : bool
                Whether to use internal self.seed_stream (for determinism)

        Returns:
            np.ndarray
                Averaged predicted class probabilities
        """
        probas = []
        for _ in range(mc_samples):
            if use_seed and self.seed_stream is not None:
                tf.random.set_seed(self.seed_stream())
            logits = self.model(X_test, training=False).numpy()
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            probas.append(probs)
        return np.mean(probas, axis=0)

    def predict_classes(self, X_test):
        """
        Returns predicted class labels (not indices).
        Handles non-zero-indexed or string class labels if provided.
        """
        logits = self.predict_logits(X_test)
        class_indices = np.argmax(logits, axis=1)
        return [self.index_to_label[int(i)] for i in class_indices]

    def evaluate_accuracy(self, X_test, y_test):
        """
        Computes accuracy using true labels and predicted labels.
        Assumes y_test contains actual class labels (not necessarily zero-indexed).
        """
        y_pred = self.predict_classes(X_test)
        if not isinstance(y_test[0], str):
            y_test = [self.index_to_label[int(i)] for i in y_test]

        return accuracy_score(y_test, y_pred)