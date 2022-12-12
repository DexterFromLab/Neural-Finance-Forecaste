from array import array
from typing import List, Tuple

import numpy as np
# def prepareTrainData(inputSize: int, dataToSlace: List[float]) -> Tuple[array, array]:
#     slices = []
#     tested_predictions = []
#
#     numberOfSlice = int(len(dataToSlace) / inputSize)
#     for slice_index in range(numberOfSlice):
#         if slice_index * inputSize + 1 >= len(dataToSlace):
#             break
#         slices.append(dataToSlace[slice_index * inputSize: (slice_index + 1) * inputSize])
#         tested_predictions.append(dataToSlace[slice_index + 1 * inputSize + 1])
#
#     return np.array(slices), np.array(tested_predictions)
#

def prepareTrainData(input_window_size: int, output_window_size: int, dataToSlice: List[float]) -> Tuple[array, array]:
    input_window = []
    output_window = []
    for i in range(len(dataToSlice)):
        if i >= input_window_size | i <= len(dataToSlice) - output_window_size:
            input_window.append(dataToSlice[i - input_window_size:i])
            output_window.append(dataToSlice[i:i + output_window_size])
    return np.array(input_window), np.array(output_window)
