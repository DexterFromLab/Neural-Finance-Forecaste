import tensorflow as tf


def createModel():
    # Tworzenie warstwy wejściowej
    input_layer = tf.keras.layers.Input(shape=(10,))

    # Tworzenie pierwszej warstwy ukrytej
    hidden_layer_1 = tf.keras.layers.Dense(units=32, activation="relu")(input_layer)

    # Tworzenie drugiej warstwy ukrytej
    hidden_layer_2 = tf.keras.layers.Dense(units=64, activation="relu")(hidden_layer_1)

    # Tworzenie warstwy wyjściowej
    output_layer = tf.keras.layers.Dense(units=1, activation="linear")(hidden_layer_2)

    # Tworzenie modelu sieci neuronowej
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss="mean_squared_error", optimizer="adam")

    return model
