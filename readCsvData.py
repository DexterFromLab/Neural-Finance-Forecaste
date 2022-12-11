import csv
from array import array
from datetime import datetime
from typing import Tuple

import numpy as np


def load_data(file_name: str) -> tuple[object, object]:
    data = []
    values = []
    with open(file_name) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Pobranie daty z kolumny "Data" i przekształcenie jej na obiekt datetime
            date = datetime.strptime(row['Data'], '%Y-%m-%d')
            # Pobranie wartości z kolumny "Zamkniecie" i zapisanie jej w tablicy
            value = float(row['Zamkniecie'])
            data.append(date.timestamp())
            values.append(value)

    # Zwrócenie danych w formacie właściwym dla modelu sieci neuronowej
    return np.array(data).tolist(), np.array(values).tolist()
