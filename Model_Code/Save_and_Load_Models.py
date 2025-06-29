import os
import json
import tensorflow as tf
from Model_Code.ConvolutionalBNN_Model import ConvolutionalBNN
import tensorflow_probability as tfp
import numpy as np
import random

def save_bnn_model(bnn, model_name, base_folder="Saved_Models", seed=42):
    """
    Saves Modelweights, Metadata and Seed a Subfolder.
    """

    model_folder = os.path.join(base_folder, model_name)
    os.makedirs(model_folder, exist_ok=True)

    weights_path = os.path.join(model_folder, f"{model_name}_weights.h5")
    metadata_path = os.path.join(model_folder, f"{model_name}_metadata.json")

    bnn.model.save_weights(weights_path)

    metadata = {
        "input_shape": tuple(int(x) for x in bnn.input_shape),
        "num_classes": int(bnn.num_classes),
        "class_labels": [str(x) for x in bnn.class_labels],
        "seed": seed
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"✅ Modell '{model_name}' saved in:\n📁 {model_folder}")

def load_bnn_model(model_name, len_x_train, base_folder="Saved_Models"):
    """
    Loads Modelweights, Metadata and Seed: Saved_Models/{model_name}/
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

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    bnn = ConvolutionalBNN(
        input_shape=input_shape,
        num_classes=num_classes,
        class_labels=class_labels,
        len_x_train=len_x_train
    )

    dummy_input = tf.zeros((1, *input_shape))
    _ = bnn.model(dummy_input)

    bnn.model.load_weights(weights_path)

    bnn.seed_stream = tfp.util.SeedStream(seed=seed, salt="flipout")

    print(f"✅ Model '{model_name}' successfully loaded with Seed {seed}: {model_folder}")
    return bnn
