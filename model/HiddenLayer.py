import tensorflow as tf
import tensorflow.keras as tfk

from model.ActivationFunction import ActivationFunction


class HiddenLayer(tfk.layers.Layer):
    def __init__(self, hidden_layer_size: int, activation: ActivationFunction):
        super(HiddenLayer, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation

    def build(self, input_shape):
        self.hidden_weights = self.add_weight(
            shape=(input_shape[-1], self.hidden_layer_size),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.hidden_layer_size,),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.hidden_weights) + self.bias)