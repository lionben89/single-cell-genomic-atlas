import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    """
    A block that applies: Linear -> BatchNorm -> ReLU -> (optional Dropout)
    """
    def __init__(self, in_dim, out_dim, dropout_prob=0.0, batchnorm_layer_class = nn.BatchNorm1d):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.batchnorm_layer_class = batchnorm_layer_class
        if batchnorm_layer_class is not None:
            self.bn = batchnorm_layer_class(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.linear(x)
        if self.batchnorm_layer_class is not None:
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
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout_prob=0.0, batchnorm_layer_class=nn.BatchNorm1d):
        super(Encoder, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(DenseBlock(prev_dim, h_dim, dropout_prob,batchnorm_layer_class))
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
    def __init__(self, latent_dim, output_dim, hidden_dims, dropout_prob=0.0,batchnorm_layer_class=nn.BatchNorm1d):
        super(Decoder, self).__init__()
        layers = []
        rev_hidden_dims = list(reversed(hidden_dims))
        prev_dim = latent_dim
        for h_dim in rev_hidden_dims:
            layers.append(DenseBlock(prev_dim, h_dim, dropout_prob,batchnorm_layer_class))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softplus())  # Suitable for count-like data.
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)

############################
# Adaptive BatchNorm Layer
############################
class AdaptiveBatchNorm1d(nn.Module):
    """
    Adaptive Batch Normalization layer that computes the statistics (mean, variance)
    on the incoming batch during the forward pass. This allows the layer to adapt to
    new data distributions rather than relying on fixed running estimates.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(AdaptiveBatchNorm1d, self).__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        # Always compute mean and variance from the current batch.
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_norm = self.gamma * x_norm + self.beta
        return x_norm

class Autoencoder(nn.Module):
    """
    Combines the Encoder and Decoder.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout_prob=0.0,batchnorm_layer_class=nn.BatchNorm1d):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims, dropout_prob,batchnorm_layer_class)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims, dropout_prob,batchnorm_layer_class)
        self.recon_loss_fn = nn.MSELoss()  # Reconstruction loss; you could use other losses.
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
    
    def loss(self, x, y=None):
        """
        Compute total loss as a combination of reconstruction loss and classification loss.
        Args:
            x: input tensor.
            true_labels: ground-truth cell type labels (as class indices) for classification.
        Returns:
            total_loss: weighted sum of reconstruction and classification loss.
        """
        recon, z = self.forward(x)
        total_loss = self.recon_loss_fn(recon, x)
        return {"total_loss":total_loss}

class AdaptiveBNAutoencoder(Autoencoder):
    """
    Autoencoder model that incorporates an Adaptive Batch Normalization (AdaBN) layer 
    after encoding the input and before decoding. The AdaBN layer recalculates the
    batch statistics on the latent representation to better adapt to new data
    distributions, which can help minimize batch effects.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout_prob=0.0,batchnorm_layer_class=nn.BatchNorm1d):
        super(AdaptiveBNAutoencoder, self).__init__(input_dim, latent_dim, hidden_dims, dropout_prob,batchnorm_layer_class)
        # Insert the AdaBN layer on the latent representation.
        self.adabn = AdaptiveBatchNorm1d(input_dim, eps=1e-5, affine=True)

    def forward(self, x):
        # Apply adaptive batch normalization on the latent representation.
        x_adapt = self.adabn(x)
        
        # Encode input
        z = self.encoder(x_adapt)
        
        # Decode from the adaptive normalized latent space.
        recon = self.decoder(z)
        return recon, z
       
class BatchAwareAutoencoder(Autoencoder):
    """
    Autoencoder with a second encoder to predict batch.
    The latent from the main encoder and the batch encoder are concatenated and
    passed to the decoder. A classification head on the batch encoder's latent space
    predicts the batch label.
    """
    def __init__(self, input_dim, latent_dim, batch_latent_dim, hidden_dims, n_classes, 
                 dropout_prob=0.0, batchnorm_layer_class=nn.BatchNorm1d, class_loss_weight=0.5):
        """
        Args:
            input_dim (int): Input dimension.
            latent_dim (int): Dimension of the main encoder's latent representation.
            batch_latent_dim (int): Dimension of the batch encoder's latent representation.
            hidden_dims (list): List of hidden layer sizes.
            n_batches (int): Number of unique batches (for classification).
            dropout_prob (float): Dropout probability.
            batchnorm_layer_class: Batch normalization layer class.
            class_loss_weight (float): Weight for the batch classification loss.
        """
        super(BatchAwareAutoencoder, self).__init__(input_dim, latent_dim, hidden_dims, dropout_prob,batchnorm_layer_class)
        
        # Second encoder for batch prediction.
        self.batch_encoder = Encoder(input_dim, batch_latent_dim, hidden_dims, dropout_prob, batchnorm_layer_class)
        
        # Decoder now takes a concatenated latent vector.
        combined_latent_dim = latent_dim + batch_latent_dim
        self.decoder = Decoder(combined_latent_dim, input_dim, hidden_dims, dropout_prob, batchnorm_layer_class)
        
        # Batch classifier: predicts batch label from batch encoder's latent vector.
        self.batch_classifier = nn.Linear(batch_latent_dim, n_classes)
        
        # Loss functions.
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.class_loss_weight = class_loss_weight

    def forward(self, x):
        # Main latent representation.
        z = self.encoder(x)
        # Batch-specific latent representation.
        z_batch = self.batch_encoder(x)
        # Predict batch logits from the batch encoder's latent.
        batch_logits = self.batch_classifier(z_batch)
        # Concatenate the two latent representations.
        z_combined = torch.cat([z, z_batch], dim=1)
        # Decode the combined latent representation.
        recon = self.decoder(z_combined)
        return recon, z, z_batch, batch_logits

    def loss(self, x, y=None):
        """
        Compute total loss as a combination of reconstruction loss and batch classification loss.
        
        Args:
            x: Input tensor.
            batch_labels: Tensor of true batch labels (as integers) for each sample.
        
        Returns:
            total_loss: Weighted sum of reconstruction and batch classification loss.
        """
        recon, z, z_batch, batch_logits = self.forward(x)
        recon_loss = self.recon_loss_fn(recon, x)
        batch_class_loss = self.class_loss_fn(batch_logits, y[:,1])
        total_loss = recon_loss + self.class_loss_weight * batch_class_loss
        return {"total_loss":total_loss, "batch_class_loss":batch_class_loss}   
    
class SupervisedBatchAwareAutoencoder(BatchAwareAutoencoder):
    """
    Extends the Autoencoder by adding a supervised cell type classification branch.
    The classification head is a simple linear layer mapping from latent space to n_classes.
    The final loss will be a combination of the reconstruction loss and classification loss.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims, n_classes, batch_latent_dim, batch_n_classes, dropout_prob=0.0, class_loss_weight=0.5,  batchnorm_layer_class=nn.BatchNorm1d, batch_class_loss_weight=0.1):
        super().__init__(input_dim,latent_dim, batch_latent_dim, hidden_dims, batch_n_classes, dropout_prob, batchnorm_layer_class, batch_class_loss_weight)
        self.classifier = nn.Linear(latent_dim+batch_latent_dim, n_classes)
        self.class_loss_fn = nn.CrossEntropyLoss()  # Classification loss for cell types
        self.class_loss_weight = class_loss_weight

    def forward(self, x):
        """
        Returns:
            recon: the reconstructed input,
            z: latent representation,
            logits: raw classification scores for cell types.
        """
        recon, z, z_batch, _ = super().forward(x)
        # Concatenate the two latent representations.
        z_combined = torch.cat([z, z_batch], dim=1)
        logits = self.classifier(z_combined)
        return recon, z, logits

    def loss(self, x, y=None):
        """
        Compute total loss as a combination of reconstruction loss and classification loss.
        Args:
            x: input tensor.
            true_labels: ground-truth cell type labels (as class indices) for classification.
        Returns:
            total_loss: weighted sum of reconstruction and classification loss.
        """
        # Use BatchAwareAutoencoder.forward directly, bypassing the override.
        recon, z, z_batch, batch_logits = BatchAwareAutoencoder.forward(self, x)
        recon_loss = self.recon_loss_fn(recon, x)
        batch_class_loss = self.class_loss_fn(batch_logits, y[:,1])
        parent_loss = {"total_loss": recon_loss + self.class_loss_weight * batch_class_loss,
                    "batch_class_loss": batch_class_loss}
        
        # Now get the cell type classification output (from the supervised forward).
        recon, z, logits = self.forward(x)
        class_loss = self.class_loss_fn(logits, y[:,0])
        total_loss = parent_loss["total_loss"] + self.class_loss_weight * class_loss
        return {"total_loss": total_loss,
                "reconstruction_loss":recon_loss,
                "signal_class_loss": class_loss,
                "batch_class_loss": parent_loss["batch_class_loss"]}
    