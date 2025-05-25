from imports import *
from ConvolutionalBNN_Model import ConvolutionalBNN

class DenseBNN(ConvolutionalBNN):
    def _build_model(self):
        kl = lambda q, p, _: tfp.distributions.kl_divergence(q, p) * self.kl_weight

        model = tfk.Sequential([
            tfk.layers.InputLayer(input_shape=self.input_shape),

            tfpl.DenseFlipout(512, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.3),

            tfpl.DenseFlipout(256, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.2),

            tfpl.DenseFlipout(128, activation='relu', kernel_divergence_fn=kl),
            tfk.layers.Dropout(0.2),

            tfpl.DenseFlipout(self.num_classes, activation=None, kernel_divergence_fn=kl)
        ])
        return model