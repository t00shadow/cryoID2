from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import estimate_bandwidth

def mean_shift_clustering(data, band):
    from sklearn.cluster import MeanShift
    from sklearn.preprocessing import StandardScaler

    # Standardize the data to have zero mean and unit variance
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply Mean Shift clustering
    mean_shift = MeanShift(bandwidth= band)  # Adjust the bandwidth parameter as needed
    mean_shift.fit(data_scaled)

    # Retrieve the cluster centers and labels
    cluster_centers = mean_shift.cluster_centers_
    labels = mean_shift.labels_

    # Convert cluster centers back to the original scale
    cluster_centers_original = scaler.inverse_transform(cluster_centers)


    return data, labels, cluster_centers_original

def calculate_weighted_center_of_mass(data, masses, labels):
    unique_labels = np.unique(labels)
    weighted_centers_of_mass = []

    for label in unique_labels:
        cluster_points = data[labels == label]
        cluster_masses = masses[labels == label]
        
        total_mass = np.sum(cluster_masses)
        weighted_sum = np.sum(cluster_points.T * cluster_masses, axis=1)
        # print(weighted_sum)
        # print(total_mass)
        if total_mass == 0:
            continue
        weighted_center_of_mass = weighted_sum / total_mass
        
        weighted_centers_of_mass.append(weighted_center_of_mass)

    return np.array(weighted_centers_of_mass)