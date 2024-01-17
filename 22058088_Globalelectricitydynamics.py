
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def read_data(file_path):
    """
    Read data from a CSV file into a Pandas DataFrame and perform initial cleanup.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    data = pd.read_csv(file_path, skiprows=4)
    return data

def drop_unnecessary_columns(data):
    """
    Drop unnecessary columns from the DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with unnecessary columns removed.
    """
    columns_to_drop = ['Country Code', 'Indicator Name', 'Indicator Code']
    data.drop(columns=columns_to_drop, axis=1, inplace=True)
    return data

def handle_missing_values(data):
    """
    Print the count of missing values in the DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.

    Returns:
    - None
    """
    print(data.isnull().sum())

def create_data_subset(data):
    """
    Create a subset of the DataFrame with selected columns.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Subset of the input DataFrame.
    """
    years = [str(year) for year in range(2000, 2023)]
    data_subset = data[['Country Name'] + years]
    return data_subset

def prepare_data_for_clustering(data_subset):
    """
    Prepare data for clustering by handling missing values and selecting relevant columns.

    Parameters:
    - data_subset (pd.DataFrame): Subset of the original DataFrame.

    Returns:
    - pd.DataFrame: Processed data ready for clustering.
    """
    data_for_clustering = data_subset.drop('Country Name', axis=1)
    data_for_clustering.fillna(0, inplace=True)
    return data_for_clustering

def find_optimal_clusters(data_for_clustering):
    """
    Find the optimal number of clusters using the Elbow Method.

    Parameters:
    - data_for_clustering (pd.DataFrame): Processed data for clustering.

    Returns:
    - None
    """
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_for_clustering)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')  # Within Cluster Sum of Squares
    plt.show()

def perform_kmeans_clustering(data_for_clustering, num_clusters=3):
    """
    Perform KMeans clustering on the processed data.

    Parameters:
    - data_for_clustering (pd.DataFrame): Processed data for clustering.
    - num_clusters (int): Number of clusters to create.

    Returns:
    - np.ndarray: Cluster labels.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data_for_clustering)
    return cluster_labels

def reduce_dimensions_with_pca(data_for_clustering):
    """
    Reduce data dimensions to 2 using PCA.

    Parameters:
    - data_for_clustering (pd.DataFrame): Processed data for clustering.

    Returns:
    - np.ndarray: Transformed data.
    """
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_for_clustering)
    return data_pca

def plot_kmeans_clustering(data_pca, cluster_labels):
    """
    Plot the results of KMeans clustering.

    Parameters:
    - data_pca (np.ndarray): Transformed data using PCA.
    - cluster_labels (np.ndarray): Cluster labels.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='black')
    plt.title('KMeans Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

def plot_scatter_comparison(data_subset, year1, year2, cluster_labels):
    """
    Plot a scatter comparison for two selected years.

    Parameters:
    - data_subset (pd.DataFrame): Subset of the original DataFrame.
    - year1 (str): First selected year.
    - year2 (str): Second selected year.
    - cluster_labels (np.ndarray): Cluster labels.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data_subset[year1], data_subset[year2], c=cluster_labels, cmap='viridis', marker='o', edgecolor='black')
    plt.title(f'KMeans Clustering - {year1} vs {year2}')
    plt.xlabel(year1)
    plt.ylabel(year2)
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

# Main part of the code
file_path = 'API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_6299951.csv'

# Step 1: Read data
data = read_data(file_path)

# Step 2: Drop unnecessary columns
data = drop_unnecessary_columns(data)

# Step 3: Handle missing values
handle_missing_values(data)

# Step 4: Create data subset
data_subset = create_data_subset(data)

# Step 5: Prepare data for clustering
data_for_clustering = prepare_data_for_clustering(data_subset)

# Step 6: Find optimal clusters
find_optimal_clusters(data_for_clustering)

# Step 7: Perform KMeans clustering
num_clusters = 3
cluster_labels = perform_kmeans_clustering(data_for_clustering, num_clusters)

# Step 8: Reduce dimensions with PCA and plot clustering results
data_pca = reduce_dimensions_with_pca(data_for_clustering)
plot_kmeans_clustering(data_pca, cluster_labels)

# Step 9: Plot scatter comparison for two selected years
year1 = '2000'
year2 = '2001'
plot_scatter_comparison(data_subset, year1, year2, cluster_labels)
