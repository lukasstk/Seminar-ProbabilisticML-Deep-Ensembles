import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import tensorflow_probability as tfp
tfpl = tfp.layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from BNN_Model import *

# ----------------------------
# Wine Quality – Vorbereitung & Training
# ----------------------------

# 1. Daten laden
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# 2. Optional: Qualität zu 3 Klassen zusammenfassen
def quality_to_class(q):
    if q <= 4:
        return "low"
    elif q <= 6:
        return "medium"
    else:
        return "high"

df['quality_label'] = df['quality'].apply(quality_to_class)
class_labels = ["low", "medium", "high"]

# 3. Features und Labels vorbereiten
X = df.drop(['quality', 'quality_label'], axis=1).values
y = df['quality_label'].values

# 4. Feature-Standardisierung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Trainings-/Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 6. Modell initialisieren, kompilieren, trainieren
bnn = BNN(input_shape=X_train.shape[1], num_classes=len(class_labels),
               len_x_train=len(X_train), class_labels=class_labels)

bnn.compile()
bnn.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.1)

# 7. Auswertung
acc = bnn.evaluate_accuracy(X_test, y_test)
print(f"\nTest Accuracy: {acc:.3f}")
