import copy

class EarlyStopping:
    """
    Early stopping callback to stop training when the validation loss does not improve.
    
    Attributes:
        patience (int): How many epochs to wait after the last improvement.
        min_delta (float): Minimum improvement required to reset the patience counter.
        best_val_loss (float): Best observed validation loss.
        wait (int): Counter for epochs with no improvement.
        best_model_state (dict): Copy of the best model weights.
        stop_training (bool): Flag indicating whether training should be stopped.
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.wait = 0
        self.best_model_state = None
        self.stop_training = False

    def on_epoch_end(self, epoch, current_val_loss, model):
        if current_val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = current_val_loss
            self.wait = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                print(f"Early stopping triggered at epoch {epoch+1}: "
                      f"No improvement in validation loss for {self.patience} consecutive epochs.")

    def on_train_end(self, model):
        # Restore the best model state if available.
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)