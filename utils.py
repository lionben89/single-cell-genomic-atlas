import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import torch

    
def calc_batch_effect_score(adata, column, n_pcs=50, random_state=42,pca_col="X_pca"):
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
    return torch.tensor(adata.X.toarray(), dtype=torch.float32).to(device)
