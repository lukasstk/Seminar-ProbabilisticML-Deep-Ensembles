from imports import *

class ConvolutionalBNN:
    def __init__(self, input_shape, num_classes, len_x_train,kl_weight=None, class_labels=None):
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

    def _build_model(self):
        """
        Build a convolutional BNN using Flipout layers and KL divergence regularization.
        """

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

    def fit(self, X_train, y_train, shuffle_data=True, random_state=None, validation_data=None, **kwargs):
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

        self.model.fit(X_train, y_train, validation_data=validation_data, **kwargs)

    def predict_logits(self, X_test):
        """
        Returns raw logits (before softmax) for X_test.
        Shape: (num_samples, num_classes)
        """
        return self.model(X_test, training=False).numpy()

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