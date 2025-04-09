import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    """
    A block that applies: Linear -> BatchNorm -> ReLU -> (optional Dropout)
    """
    def __init__(self, in_dim, out_dim, dropout_prob=0.0):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """
    Encoder that maps input_dim -> hidden_dims (list) -> latent_dim.
    Each hidden layer is built using a DenseBlock.
    The final layer is linear (without BN/activation) to get the latent representation.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout_prob=0.0):
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(DenseBlock(prev_dim, h_dim, dropout_prob))
            prev_dim = h_dim
        # Final layer: linear projection to latent space.
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decoder that mirrors the encoder.
    It maps latent_dim -> reversed(hidden_dims) -> output_dim.
    The final layer uses Softplus to ensure non-negative outputs.
    """
    def __init__(self, latent_dim, output_dim, hidden_dims, dropout_prob=0.0):
        super(Decoder, self).__init__()
        layers = []
        rev_hidden_dims = list(reversed(hidden_dims))
        prev_dim = latent_dim
        for h_dim in rev_hidden_dims:
            layers.append(DenseBlock(prev_dim, h_dim, dropout_prob))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softplus())  # Suitable for count-like data.
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    """
    Combines the Encoder and Decoder.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout_prob=0.0):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims, dropout_prob)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims, dropout_prob)
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
