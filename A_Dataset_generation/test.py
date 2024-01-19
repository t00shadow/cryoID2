import json
import os

import numpy as np

flattened_data = np.array([1, 2, 3, 4, 5, 2, 1, 3, 4])
counts = np.bincount(flattened_data)


array1 = np.array([1, 2, 3])
array2 = np.array([4, 5])

result_array = array1 + array2
print(result_array)
