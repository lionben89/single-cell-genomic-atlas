import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import convert_to_dense_torch

def get_dataloader(adata, batch_size, device):
    # Convert the expression matrix to a PyTorch tensor.
    X = convert_to_dense_torch(adata, device)
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader


def train_model(model, dataloader, epochs, lr, device):
    """
    Train the autoencoder.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # Reconstruction loss; you could use other losses.
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    return model