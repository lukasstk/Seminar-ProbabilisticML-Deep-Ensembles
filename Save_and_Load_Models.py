import os
import json
import tensorflow as tf
from ConvolutionalBNN_Model import ConvolutionalBNN
import tensorflow_probability as tfp
import numpy as np
import random

def save_bnn_model(bnn, model_name, base_folder="Saved_Models", seed=42):
    """
    Speichert Modellgewichte, Metadaten und Seed in einem automatisch erzeugten Unterordner.
    """

    model_folder = os.path.join(base_folder, model_name)
    os.makedirs(model_folder, exist_ok=True)

    weights_path = os.path.join(model_folder, f"{model_name}_weights.h5")
    metadata_path = os.path.join(model_folder, f"{model_name}_metadata.json")

    # Gewichte speichern
    bnn.model.save_weights(weights_path)

    # Metadaten inkl. Seed speichern
    metadata = {
        "input_shape": tuple(int(x) for x in bnn.input_shape),
        "num_classes": int(bnn.num_classes),
        "class_labels": [str(x) for x in bnn.class_labels],
        "seed": seed
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"✅ Modell '{model_name}' gespeichert unter:\n📁 {model_folder}")

def load_bnn_model(model_name, len_x_train, base_folder="Saved_Models"):
    """
    Lädt Modellgewichte, Metadaten und Seed aus: Saved_Models/{model_name}/
    """

    model_folder = os.path.join(base_folder, model_name)
    weights_path = os.path.join(model_folder, f"{model_name}_weights.h5")
    metadata_path = os.path.join(model_folder, f"{model_name}_metadata.json")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    input_shape = tuple(metadata["input_shape"])
    num_classes = metadata["num_classes"]
    class_labels = metadata["class_labels"]
    seed = metadata.get("seed", 42)

    # Globalen Seed setzen
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Modell rekonstruieren
    bnn = ConvolutionalBNN(
        input_shape=input_shape,
        num_classes=num_classes,
        class_labels=class_labels,
        len_x_train=len_x_train
    )

    # Dummy Input zur Initialisierung
    dummy_input = tf.zeros((1, *input_shape))
    _ = bnn.model(dummy_input)

    # Gewichte laden
    bnn.model.load_weights(weights_path)

    # SeedStream für deterministisches Sampling anhängen
    bnn.seed_stream = tfp.util.SeedStream(seed=seed, salt="flipout")

    print(f"✅ Modell '{model_name}' erfolgreich geladen mit Seed {seed} aus: {model_folder}")
    return bnn
