import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import mode
import numpy as np

def calc_lisi_batch_score(adata, column, pca_col="X_pca", k=30):
    """
    Computes the Local Inverse Simpson’s Index (LISI) for each cell from an AnnData object,
    using the specified low-dimensional embedding and batch labels.
    
    The LISI score ranges from 1 (no mixing) to M (perfect mixing), where M is the total number of distinct groups (e.g., batches)
    available in the neighborhood. For example, in a dataset with two batches, 
    the LISI score will lie between 1 and 2, and with three batches, between 1 and 3.

    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object containing single-cell data.
    column : str
        Column name in adata.obs that contains batch labels.
    pca_col : str, optional (default: "X_pca")
        Key in adata.obsm to use as the low-dimensional embedding (e.g., "X_pca", "X_umap").
    k : int, optional (default: 30)
        Number of nearest neighbors to consider (excluding the cell itself).

    Returns:
    --------
    lisi_values : np.ndarray
        Array of the LISI value computed for each cell.
    mean_lisi : float
        The mean LISI value across all cells.
    """

    # Check if the specified embedding exists.
    if pca_col not in adata.obsm.keys():
        raise KeyError(f"Embedding '{pca_col}' not found in adata.obsm.")
    
    # Check if the specified batch key exists.
    if column not in adata.obs.columns:
        raise KeyError(f"Batch key '{column}' not found in adata.obs.")
    
    # Retrieve the embedding and batch labels.
    data = adata.obsm[pca_col]
    labels = adata.obs[column].values
    
    n_cells = data.shape[0]
    
    # Fit nearest neighbors to the data.
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Exclude the cell itself (first neighbor).
    neighbor_indices = indices[:, 1:]
    
    # Initialize array to hold LISI values for each cell.
    lisi_values = np.zeros(n_cells)
    
    for i in range(n_cells):
        # Get the labels of the k nearest neighbors for cell i.
        neighbor_labels = labels[neighbor_indices[i]]
        # Count occurrences of each unique label.
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        # Calculate proportions for each label.
        proportions = counts / k
        # Compute Simpson's diversity index.
        simpson_index = np.sum(proportions ** 2)
        # LISI is defined as the inverse of Simpson's index.
        lisi_values[i] = 1.0 / simpson_index

    mean_lisi = np.mean(lisi_values)
    print(f"LISI score '{column}': {mean_lisi:.4f}")
    return lisi_values, mean_lisi

    
def calc_ari_batch_score(adata, column, n_pcs=50, random_state=42,pca_col="X_pca"):
    """
    Estimate batch effect by clustering PCA and comparing to a metadata column using ARI.

    Parameters:
        adata        : AnnData object with PCA computed (or raw, PCA will be run)
        column       : str, column in adata.obs to compare against (e.g. 'assay')
        n_pcs        : int, number of PCs to use
        random_state : for reproducibility

    Returns:
        ARI score between KMeans clustering and the batch labels.
    """
    # Check if column exists
    if column not in adata.obs.columns:
        raise ValueError(f"'{column}' not found in adata.obs")

    # Run PCA if not done yet
    if pca_col not in adata.obsm:
        from scanpy.pp import pca
        print("PCA not found — running PCA now...")
        sc.pp.pca(adata, n_comps=n_pcs)
    
    
    X = adata.obsm[pca_col][:, :n_pcs]

    # Number of batch categories
    n_clusters = adata.obs[column].nunique()

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X)

    # Compare KMeans clusters to true batch labels
    true_labels = adata.obs[column].astype(str).values
    ari = adjusted_rand_score(true_labels, cluster_labels)

    print(f"ARI between KMeans clusters and '{column}': {ari:.4f}")
    return ari    

def convert_to_dense_torch(adata, device):
    return torch.tensor(adata.X, dtype=torch.float32).to(device)

def plot_knn_balanced_accuracy_by_group(adata, knn_column, true_label_col, group_col, n_neighbors=5, metric='euclidean', 
                                          title='kNN Balanced Accuracy by Group', figsize=(10,6)):
    """
    For each cell in an AnnData object, compute a kNN prediction based on a specified
    column in adata.obs. Then, group cells by a group column and plot the balanced accuracy
    (balanced_accuracy_score) of these predictions for each group.
    
    Args:
        adata: AnnData object with observations in adata.obs.
        knn_column (str): Column name (from adata.obs) used to compute the kNN graph.
                           The values in this column should be numeric (or castable to float).
        true_label_col (str): Column name in adata.obs containing the true class labels.
        group_col (str): Column name in adata.obs defining groups (e.g., batches or cell types)
                         for which to compute balanced accuracy.
        n_neighbors (int): Number of neighbors to use for prediction (excluding self). Default is 5.
        metric (str): Distance metric for kNN (default 'euclidean').
        title (str): Title of the bar plot.
        figsize (tuple): Figure size for the plot.
    
    Returns:
        group_bal_accuracy (dict): A dictionary mapping each group (unique value in group_col)
                                   to its corresponding balanced accuracy score.
    """
    # Extract values from the specified column to compute kNN.
    # Reshape to 2D (n_samples, 1) so each sample is a point in 1D space.
    X = adata.obs[[knn_column]].values.astype(float)
    num_samples = X.shape[0]
    
    # Typically the closest neighbor is the sample itself.
    # So we request (n_neighbors + 1) and then remove the self neighbor.
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Remove the self index from the neighbor list.
    # Here, for each sample, we remove any neighbor with the same index as the sample.
    corrected_indices = []
    for i in range(num_samples):
        neigh = indices[i]
        # Remove self: often self is the very first neighbor.
        neigh = neigh[neigh != i]
        # In case we have extra neighbors, only retain the first n_neighbors.
        corrected_indices.append(neigh[:n_neighbors])
    corrected_indices = np.stack(corrected_indices, axis=0)
    
    # Retrieve the true labels from the specified column.
    true_labels = adata.obs[true_label_col].values
    
    # Predict the label for each sample by majority vote among its neighbors' true labels.
    predicted_labels = []
    for i in range(num_samples):
        neighbor_indices = corrected_indices[i]
        neighbor_labels = true_labels[neighbor_indices]
        # Use mode from scipy to compute the majority vote.
        # In case of ties, mode() returns the smallest value.
        pred = mode(neighbor_labels).mode[0]
        predicted_labels.append(pred)
    predicted_labels = np.array(predicted_labels)
    
    # Compute balanced accuracy per group.
    groups = adata.obs[group_col].unique()
    group_bal_accuracy = {}
    for grp in groups:
        mask = adata.obs[group_col] == grp
        grp_true = true_labels[mask]
        grp_pred = predicted_labels[mask]
        score = balanced_accuracy_score(grp_true, grp_pred)
        group_bal_accuracy[grp] = score
    
    # Plot the balanced accuracy per group.
    plt.figure(figsize=figsize)
    plt.bar(list(group_bal_accuracy.keys()), list(group_bal_accuracy.values()))
    plt.xlabel("Group")
    plt.ylabel("Balanced Accuracy")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
    
    return group_bal_accuracy
