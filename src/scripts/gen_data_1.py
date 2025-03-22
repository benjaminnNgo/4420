import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_data(num_clusters=1000, num_samples=5000, dimensions=12, space_bounds=(0, 10), deviation=1):
    # Generate random cluster centers within the given space bounds
    cluster_centers = np.random.uniform(space_bounds[0], space_bounds[1], size=(num_clusters, dimensions))
    
    # Generate data points by sampling from Gaussian distributions centered at these clusters
    samples_per_cluster = num_samples // num_clusters
    data = []
    for center in cluster_centers:
        cluster_samples = np.random.normal(loc=center, scale=deviation, size=(samples_per_cluster, dimensions))
        data.append(cluster_samples)
    
    data = np.vstack(data)
    return data

# Generate data
data = generate_gaussian_data()

# Print shape to verify
print("Generated Data Shape:", data.shape)

# Visualizing the first two dimensions of the data
plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Scatter plot of first two dimensions")
plt.show()
