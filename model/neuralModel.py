from typing import List

import tensorflow as tf

from model.HiddenLayer import HiddenLayer


def createModel(input_size: int, output_size: int, hidden_layers: List[HiddenLayer]):
    # Tworzenie warstwy wejściowej
    input_layer = tf.keras.layers.Input(shape=(input_size,))

    # Tworzenie warstw ukrytych
    hidden_layer = input_layer
    for layer in hidden_layers:
        hidden_layer = layer(hidden_layer)

    # Tworzenie warstwy wyjściowej
    output_layer = tf.keras.layers.Dense(units=output_size, activation="linear")(hidden_layer)

    # Tworzenie modelu sieci neuronowej
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model

