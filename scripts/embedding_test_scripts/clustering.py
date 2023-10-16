import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Generating random data for Class A and Class B
np.random.seed(42)  # for reproducibility
num_samples = 1000  # Number of samples for each class
num_features = 2000  # Number of features
num_clusters = 2  # Number of clusters (Class A and Class B)

# Creating random data for Class A and Class B
class_A_data = np.random.rand(num_samples, num_features)
class_B_data = np.random.rand(num_samples, num_features)

# Combining the data
combined_data = np.vstack((class_A_data, class_B_data))

# Applying K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(combined_data)

# Creating a DataFrame with cluster labels
cluster_df = pd.DataFrame({'Cluster': cluster_labels})

# Adding class labels (0 for Class A, 1 for Class B)
class_labels = np.array([0] * num_samples + [1] * num_samples)
cluster_df['Class'] = class_labels

# Grouping by cluster and calculating the average class label for each cluster
cluster_summary = cluster_df.groupby('Cluster')['Class'].mean().reset_index()

# Printing the cluster summary
print(cluster_summary)