# Import bibliotek
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from model import neuralModel
from model.ActivationFunction import ActivationFunction
from model.HiddenLayer import HiddenLayer
from utils import readCsvData, sliceData
import numpy as np
from deap.tools import cxTwoPoint

def neural_network_training_with_mse(input_layer_size: int, output_layer_size: int, first_layer_size: int,
                                     second_layer_size: int, train_epochs, hidden_layers: List[HiddenLayer]) -> float:
    # Zdefiniowanie wielkości wektora na wejściu i wyjściu sieci

    # Wczytanie danych historycznych
    dates, values = readCsvData.load_data("../resources/wig20_d_lasts.csv")

    # Ustandaryzowanie danych
    scaler = MinMaxScaler()
    values = np.array(values).reshape(-1, 1)
    # Skalowanie danych
    scaled_data = scaler.fit_transform(values).flatten().tolist()

    # Przygotowanie danych do nauki i testowania 10 odczytów gieldowych na wejście i 11-sty jako przewidywany wynik
    scaled_data_input, scaled_data_output = sliceData.prepareTrainData(input_layer_size, output_layer_size, scaled_data)

    scaled_train_data_input, scaled_test_data_input = train_test_split(scaled_data_input, test_size=0.2, shuffle=False)
    scaled_train_data_output, scaled_test_data_output = train_test_split(scaled_data_output, test_size=0.2,
                                                                         shuffle=False)

    # Kompilacja modelu
    model = neuralModel.createModel(input_layer_size, output_layer_size, hidden_layers)

    # Uczenie sieci
    history = model.fit(scaled_train_data_input, scaled_train_data_output, epochs=train_epochs, batch_size=1, verbose=2)

    # Ocena dokładności na danych testowych
    scaled_predicted_data_output = model.predict(scaled_test_data_input)
    predicted_data_output = scaler.inverse_transform(np.array(scaled_predicted_data_output))
    test_data_output = scaler.inverse_transform(np.array(scaled_test_data_output))
    for test_data_index in range(len(predicted_data_output)):
        print(
            f"Przewidywany wynik: {predicted_data_output[test_data_index]}, dane historyczne: {test_data_output[test_data_index]}\n")

    # Predykcja najbliższej wartości
    new_scalled_predicted_value = model.predict(np.array(scaled_data[-input_layer_size:]).reshape(1, -1))
    print(f"Ostatnia wartość historyczna: {scaler.inverse_transform(np.array(scaled_data[-1:]).reshape(-1, 1))}")
    new_predicted_value = scaler.inverse_transform(new_scalled_predicted_value)
    print(f"Najbliższa przewidywana wartość: {new_predicted_value}")

    # Obliczamy MSE
    mse_sum = 0
    for i in range(len(predicted_data_output)):
        mse = np.mean((np.array(test_data_output[i]) - np.array(predicted_data_output[i])) ** 2)
        mse_sum += mse

    print(f"Suma MSE dla testów: {mse_sum}")

    return mse_sum


def main():
    results = []

    hidden_layers = [HiddenLayer(hidden_layer_size=32, activation=ActivationFunction.RELU),
                     HiddenLayer(hidden_layer_size=32, activation=ActivationFunction.TANH)]

    results.append(neural_network_training_with_mse(100, 10, 32, 32, 1, hidden_layers))
    results.append(neural_network_training_with_mse(100, 10, 32, 32, 10, hidden_layers))
    results.append(neural_network_training_with_mse(100, 10, 32, 32, 100, hidden_layers))

    for i in range(len(results)):
        print(f"Wynik dla sieci: {results[i]}")


if __name__ == "__main__":
    main()
