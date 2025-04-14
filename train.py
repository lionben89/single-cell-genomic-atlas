import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Sampler


from utils import convert_to_dense_torch

# Custom sampler that yields batches composed of samples all in the same group.
class GroupBatchSampler(Sampler):
    def __init__(self, indices, groups, batch_size, shuffle=True):
        """
        Args:
            indices (list): List of indices for all samples.
            groups (array-like): Group label corresponding to each index.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle indices within each group and the order of batches.
        """
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Build a mapping: group -> list of indices in that group
        self.group_to_indices = {}
        for idx, group in zip(indices, groups):
            self.group_to_indices.setdefault(group, []).append(idx)

    def __iter__(self):
        batches = []
        # Process each group separately
        for group, idxs in self.group_to_indices.items():
            if self.shuffle:
                random.shuffle(idxs)
            # Partition indices of the group into batches
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i+self.batch_size]
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        total_batches = 0
        for idxs in self.group_to_indices.values():
            total_batches += (len(idxs) + self.batch_size - 1) // self.batch_size
        return total_batches

def get_dataloader(adata, label_columns, batch_size, device, shuffle=True, group_batch=False, group_by=None):
    """
    Creates a DataLoader for the input data.
    
    Args:
        adata: Anndata object containing the expression matrix and observation metadata.
        label_columns: Column(s) in adata.obs used as labels.
        batch_size (int): Desired batch size.
        device: Device on which tensors should be allocated.
        shuffle (bool, optional): Whether to shuffle the dataset (default: True).
        group_batch (column, optional): If True, each batch will be drawn from a single group.
        group_by (str, optional): Column name from adata.obs to group by. If None and group_batch is True, 
                                  the first label column is used.
    
    Returns:
        DataLoader object.
    """
    # Convert the expression matrix to a PyTorch tensor.
    print(adata.shape)
    X = convert_to_dense_torch(adata, device)
    
    # Get labels from adata.obs for the selected columns.
    labels_df = adata.obs[label_columns]
    
    # For multiple label columns, factorize non-numeric columns individually.
    labels_np = np.zeros(labels_df.shape, dtype=int)
    for i, col in enumerate(labels_df.columns):
        col_data = labels_df[col]
        if not pd.api.types.is_numeric_dtype(col_data):
            # Factorize converts the column into numeric codes.
            codes, _ = pd.factorize(col_data)
            labels_np[:, i] = codes
        else:
            labels_np[:, i] = col_data.values
    y = torch.tensor(labels_np).to(device)
    
    dataset = TensorDataset(X, y)
    
    if group_batch:
        # Determine the group for batching.
        # If group_by is provided, use that column; otherwise, default to the first label column.
        if group_by is None:
            group_data = labels_df.iloc[:, 0].values
        else:
            group_data = labels_df[group_by].values
        
        # Convert group data to numeric codes if not already numeric.
        if not pd.api.types.is_numeric_dtype(group_data):
            group_data2, _ = pd.factorize(group_data)
        
        # Create a list of indices corresponding to each sample.
        indices = list(range(len(y)))
        # Use the custom GroupBatchSampler to ensure each batch is homogeneous.
        sampler = GroupBatchSampler(indices, group_data2, batch_size, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        
    return dataloader
                           
def random_split_by_group_key(adata, label_columns, group_key, train_frac=0.7, val_frac=0.15, 
                              batch_size=64, device="cpu", shuffle=True, group_batch=False, 
                              group_by=None, return_dataloader=True):
    """
    Splits an AnnData object into train, validation, and test sets.
    
    If group_key is provided (i.e. not None), the unique groups from adata.obs[group_key]
    are randomly split so that entire groups are allocated to one of the splits.
    
    If group_key is None, a regular random split is performed at the cell level. Optionally,
    stratification is done using the first entry in label_columns.
    
    Args:
        adata: AnnData object.
        label_columns: list of column names from adata.obs (e.g., ['cell_type', 'assay']).
        group_key: Column name in adata.obs to group by (e.g. 'donor_id'). If None, regular split.
        train_frac (float): Fraction of data for training.
        val_frac (float): Fraction of data for validation.
        batch_size (int): Batch size for dataloaders.
        device (str): Device string.
        shuffle (bool): Whether to shuffle the data.
        group_batch (bool): Whether to use group-based batching.
        group_by: Additional grouping info for get_dataloader (if needed).
        return_dataloader (bool): If True, returns DataLoaders; otherwise returns AnnData objects.
    
    Returns:
        Tuple of (train, validation, test) dataloaders (or AnnData objects if return_dataloader is False).
    """
    if group_key is not None:
        s_col = group_key
    else:
        s_col = label_columns[0]
    # Regular splitting (cell-level).
    # Get the indices of all cells.
    indices = np.arange(adata.n_obs)
    # Use the first label column for stratification if available.
    stratify = adata.obs[s_col].values
    # First, split into (train+val) and test.
    train_val_idx, test_idx = train_test_split(
        indices, test_size=1 - (train_frac + val_frac), stratify=stratify, random_state=42, shuffle=shuffle
    )
    # For the (train+val) split, update stratification.
    stratify_train = adata.obs[s_col].iloc[train_val_idx].values
    # Now split the (train+val) indices into train and validation.
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size= val_frac, stratify=stratify_train, random_state=42, shuffle=shuffle
    )
    
    adata_train = copy.copy(adata[train_idx])
    adata_val = copy.copy(adata[val_idx])
    adata_test = copy.copy(adata[test_idx])
    
    if return_dataloader:
        # Assume get_dataloader is defined elsewhere.
        return get_dataloader(adata_train, label_columns, batch_size, device, shuffle, group_batch=False, group_by=group_by),\
                get_dataloader(adata_val, label_columns, batch_size, device, shuffle, group_batch=False, group_by=group_by),\
                get_dataloader(adata_test, label_columns, batch_size, device, shuffle, group_batch=False, group_by=group_by)
    else:
        return adata_train, adata_val, adata_test

def train_model(model, train_dataloader, val_dataloader, epochs, lr, device, callbacks=None):
    """
    Train the model with support for callbacks (e.g., early stopping) and multiple loss components.
    
    The model.loss(x, y) function is expected to return a dictionary where each key corresponds to
    a loss value (e.g., {"total_loss": ..., "batch_class_loss": ...}), and potentially more.
    
    Parameters:
        model           : the model to train.
        train_dataloader: DataLoader for training data.
        val_dataloader  : DataLoader for validation data.
        epochs          : maximum number of epochs.
        lr              : learning rate.
        device          : training device ("cpu" or "cuda").
        callbacks       : list of callback instances to be called at the end of each epoch.
    
    Returns:
        The model trained up to the best validation loss.
    """
    # Move model to device and set up the optimizer.
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # If no callbacks provided, use an empty list.
    if callbacks is None:
        callbacks = []
    
    for epoch in range(epochs):
        
        model.train()
        train_loss_sum = {}  # dictionary to accumulate losses over batches.
        total_train_samples = 0
        
        for batch in tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{epochs}"):
            x = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()
            loss_dict = model.loss(x, y)  # assume this returns a dictionary of loss values.
            # Backward on one designated loss. Here, we assume "total_loss" always exists.
            loss_dict["total_loss"].backward()
            optimizer.step()
            
            batch_size = x.size(0)
            total_train_samples += batch_size
            
            # Sum each loss (weight with batch size) for averaging later.
            for key, loss_value in loss_dict.items():
                train_loss_sum[key] = train_loss_sum.get(key, 0.0) + loss_value.item() * batch_size
        
        # Calculate average training losses for all keys.
        avg_train_losses = {key: total / total_train_samples for key, total in train_loss_sum.items()}
        
        # Evaluate on validation set.
        model.eval()
        val_loss_sum = {}
        total_val_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Val Epoch {epoch+1}/{epochs}"):
                x = batch[0].to(device)
                y = batch[1].to(device)
                loss_dict = model.loss(x, y)
                batch_size = x.size(0)
                total_val_samples += batch_size
                for key, loss_value in loss_dict.items():
                    val_loss_sum[key] = val_loss_sum.get(key, 0.0) + loss_value.item() * batch_size
        
        avg_val_losses = {key: total / total_val_samples for key, total in val_loss_sum.items()}
        
        # Print all losses.
        train_loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_train_losses.items()])
        val_loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_val_losses.items()])
        print(f"Epoch {epoch+1}/{epochs} - Train Losses: {train_loss_str} || Val Losses: {val_loss_str}")
        
        # Trigger callbacks.
        for cb in callbacks:
            cb.on_epoch_end(epoch, avg_val_losses.get("total_loss", None), model)
        
        # Stop if any callback signals to stop.
        if any(getattr(cb, "stop_training", False) for cb in callbacks):
            print("Stopping training due to early stopping callback.")
            break
    
    # At training end, trigger on_train_end callbacks.
    for cb in callbacks:
        if hasattr(cb, "on_train_end"):
            cb.on_train_end(model)
    
    return model

def predict_by_group_and_populate(model, adata, group_key, batch_size=64, device="cpu", prediction_key="X_pred"):
    """
    For each unique group in adata.obs[group_key], use the trained model to predict the output
    (e.g. latent representation) and then store the predictions back into the AnnData object in
    the obsm field under the specified prediction_key. This ensures that the predictions are
    computed group-wise (which may be required for domain-specific post-processing or AdaBN).

    Parameters:
        model          : Trained model
        adata          : AnnData object containing the data.
        group_key      : Column name in adata.obs by which to group the data (e.g., donor_id or batch).
        batch_size     : Batch size for processing each group (default: 64).
        device         : Device to run the computations on (default: "cpu").
        prediction_key : Key in adata.obsm where the predictions will be stored (default: "X_pred").

    Returns:
        The AnnData object with predictions stored in adata.obsm[prediction_key] in the same order as adata.obs.
    """
    model.eval()
    n_cells = adata.shape[0]
    # Prepare a placeholder list to collect predictions in original cell order.
    predictions_list = [None] * n_cells
    
    # Get the unique groups defined by group_key.
    unique_groups = pd.unique(adata.obs[group_key])
    
    for group in unique_groups:
        # Find the indices of the cells belonging to the current group.
        group_idx = np.where(adata.obs[group_key] == group)[0]
        
        # Subset the AnnData object for the current group.
        adata_group = copy.copy(adata[group_idx])
        
        # Convert the expression data of the group to a torch tensor.
        X_group = convert_to_dense_torch(adata_group, device)
        dataset = TensorDataset(X_group)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        group_predictions = []
        with torch.no_grad():
            # For each batch in the group, compute the prediction.
            for (batch,) in tqdm(dataloader, desc=f"Prediction {group}"):
                pred = model.encoder(batch)
                group_predictions.append(pred.cpu().numpy())
        
        # Concatenate predictions from all batches for the current group.
        group_predictions = np.concatenate(group_predictions, axis=0)
        
        # Assign each prediction to the corresponding cell index.
        for i, idx in enumerate(group_idx):
            predictions_list[idx] = group_predictions[i]
    
    # Convert the list of predictions into a 2D numpy array.
    predictions_array = np.array(predictions_list)
    
    # Populate the AnnData object with the predictions.
    adata.obsm[prediction_key] = predictions_array
    return adata

def agg_group_level_embeddings(adata, emb_keys, group_key="donor_id", label_key="disease"):
    """
    Aggregate group-level embeddings from an AnnData object for a given list of embedding keys.
    
    For each key in emb_keys, the function:
      1. Extracts the embedding from `adata.obsm[key]`.
      2. Creates a DataFrame with the embedding data (indexed by cell names).
      3. Adds group identifiers and a label column from `adata.obs`.
      4. Groups by the group id and computes the mean for each embedding dimension.
      5. Retrieves the group-level label (here we take the first value in each group).
    
    Args:
        adata: An AnnData object with:
            - `adata.obsm` containing embeddings (e.g., "X_pca", "X_AE").
            - `adata.obs` containing group ids (group_key) and labels (label_key).
        emb_keys (list of str): List of keys (e.g., ['X_pca', 'X_AE']) present in adata.obsm.
        group_key (str): Column name in adata.obs that indicates group membership.
        label_key (str): Column name in adata.obs that indicates the label (e.g., disease status).
    
    Returns:
        dict: A dictionary where each key is an embedding key from emb_keys and each value 
              is a DataFrame with one row per group. The DataFrame contains mean embedding 
              columns (e.g., "feat0", "feat1", …) plus a column for the group-level label.
    """
    group_emb_dict = {}
    
    # For each embedding key, process the corresponding embedding matrix.
    for key in emb_keys:
        # Create a DataFrame from the embedding matrix.
        emb_df = pd.DataFrame(adata.obsm[key], index=adata.obs_names)
        # Group by the group identifier (using the metadata in adata.obs) and compute the mean for each group.
        group_mean = emb_df.groupby(adata.obs[group_key]).mean()
        # Retrieve the group-level label (assumes all cells within a group share the same label).
        group_label = adata.obs.groupby(adata.obs[group_key])[label_key].first()
        # Append the label as a new column.
        group_mean[label_key] = group_label
        group_emb_dict[key] = group_mean
    
    return group_emb_dict