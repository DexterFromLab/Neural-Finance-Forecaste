from array import array
from typing import List, Tuple

import numpy as np


def prepareTrainData(inputSize: int, dataToSlace: List[float]) -> Tuple[array, array]:
    slices = []
    tested_predictions = []

    numberOfSlice = int(len(dataToSlace) / inputSize)
    for slice_index in range(numberOfSlice):
        if slice_index * inputSize + 1 >= len(dataToSlace):
            break
        slices.append(dataToSlace[slice_index * inputSize : (slice_index +1) * inputSize])
        tested_predictions.append(dataToSlace[slice_index +1 * inputSize + 1])

    return np.array(slices), np.array(tested_predictions)