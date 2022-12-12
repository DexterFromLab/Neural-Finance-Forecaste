import tensorflow as tf


def createModel(input_size: int, output_size: int):
    # Tworzenie warstwy wejściowej
    input_layer = tf.keras.layers.Input(shape=(input_size,))

    # Tworzenie pierwszej warstwy ukrytej
    hidden_layer_1 = tf.keras.layers.Dense(units=32, activation="relu")(input_layer)

    # Tworzenie drugiej warstwy ukrytej
    hidden_layer_2 = tf.keras.layers.Dense(units=32, activation="relu")(hidden_layer_1)

    # Tworzenie warstwy wyjściowej
    output_layer = tf.keras.layers.Dense(units=output_size, activation="linear")(hidden_layer_2)

    # Tworzenie modelu sieci neuronowej
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss="mean_squared_error", optimizer="adam")

    return model
