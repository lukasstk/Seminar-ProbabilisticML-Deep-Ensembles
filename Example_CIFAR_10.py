#%% Dataset with Integer Labels
from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN

# Set seed for reproducibility
random_state = np.random.RandomState(42)

# Load and normalize CIFAR-10 dataset
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Flatten y labels (CIFAR-10 loads them as shape (n, 1))
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Initialize model
model_single = ConvolutionalBNN(
    input_shape=(32, 32, 3),
    num_classes=10,
    len_x_train=len(X_train)
)

# Compile and train
model_single.compile()

model_single.fit(
    X_train, y_train,
    epochs=10,  # Try more epochs (e.g., 20) for better accuracy
    batch_size=128,
    verbose=1,
    validation_data =(X_val, y_val)
)

# Evaluate
acc_single = model_single.evaluate_accuracy(X_test, y_test)
print(f"BNN Single Accuracy on CIFAR-10: {acc_single:.4f}")

# Predict a few labels
model_single_predictions = model_single.predict_classes(X_test[:10])
print("Predicted integer labels:", model_single_predictions)