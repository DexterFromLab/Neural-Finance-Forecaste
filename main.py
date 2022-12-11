# Import bibliotek
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import readCsvData
import sliceData
import neuralModel

# Wczytanie danych historycznych
dates, values = readCsvData.load_data("C:\\Users\\barto\\Desktop\\dane historyczne\\wig20_d_lasts.csv")

# Ustandaryzowanie danych
scaler = sklearn.preprocessing.MinMaxScaler()
values = np.array(values).reshape(-1, 1)
# Skalowanie danych
scaled_data = scaler.fit_transform(values).flatten().tolist()

# Przygotowanie danych do nauki i testowania 10 odczytów gieldowych na wejście i 11-sty jako przewidywany wynik
scaled_data_input, scaled_data_output = sliceData.prepareTrainData(10, scaled_data)

scaled_train_data_input, scaled_test_data_input = train_test_split(scaled_data_input, test_size= 0.2, shuffle= False)
scaled_train_data_output, scaled_test_data_output = train_test_split(scaled_data_output, test_size= 0.2, shuffle= False)

# Kompilacja modelu
model = neuralModel.createModel()

# Uczenie sieci
history = model.fit(scaled_data_input, scaled_data_output, epochs=200, batch_size=1, verbose=2)

# Ocena dokładności na danych testowych
scaled_predicted_data_output = model.predict(scaled_test_data_input)
predicted_data_output = scaler.inverse_transform(np.array(scaled_predicted_data_output).reshape(-1, 1))
test_data_output = scaler.inverse_transform(np.array(scaled_test_data_output).reshape(-1, 1))
for test_data_index in range(len(predicted_data_output)):
    print(f"Przewidywany wynik: {predicted_data_output[test_data_index]}, dane historyczne: {test_data_output[test_data_index]}\n")

# Predykcja najbliższej wartości
new_scalled_predicted_value = model.predict(np.array(scaled_data[-10:]).reshape(1, -1))
print(f"Ostatnia wartość historyczna: {scaler.inverse_transform(np.array(scaled_data[-1:]).reshape(-1,1))}")
new_predicted_value = scaler.inverse_transform(new_scalled_predicted_value)
print(f"Najbliższa przewidywana wartość: {new_predicted_value}")



