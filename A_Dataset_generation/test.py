import json
import os

import numpy as np

around_points = []
extension = range(-2, 2 + 1)

for dz in extension:
    for dy in extension:
        for dx in extension:
            dist = dz**2 + dy**2 + dx**2
            if dist <= 2**2:
                around_points.append([dz, dy, dx])
around_points = np.array(around_points)
print(around_points.size)
around_points = np.vstack((around_points, around_points + np.array([0, 0, 1])))
around_points = np.vstack((around_points, around_points + np.array([0, 1, 0])))
around_points = np.vstack((around_points, around_points + np.array([1, 0, 0])))
print(around_points.size)
around_points = np.unique(around_points, axis=0).tolist

