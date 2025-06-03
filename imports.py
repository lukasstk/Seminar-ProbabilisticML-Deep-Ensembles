# Core Python utilities
import random

# Numerical computing
import numpy as np

# Machine learning and neural networks
import tensorflow as tf
import tf_keras as tfk
import tensorflow_probability as tfp
tfpl = tfp.layers

# Data preprocessing and evaluation
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import calibration_curve
from scipy.stats import entropy