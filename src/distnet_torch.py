import time
import torch
import torch.nn as nn
import torch.optim as optim


def nllh_loss_torch(y_true, y_pred):
    """
    Negative log-likelihood loss for log-normal distribution.
    y_pred: tensor of shape [batch, 2], first column s, second column scale
    y_true: tensor of shape [batch, 1]
    """
    # ensure tensors on same device
    y_true = y_true.to(y_pred.device)

    s = y_pred[:, 0:1]
    scale = y_pred[:, 1:2]

    log_scale = torch.log(scale)
    log_true = torch.log(y_true)

    help1 = log_true - log_scale
    help1 = 0.5 * ((help1 / s) ** 2)

    lh = -torch.log(s) - log_true - help1

    # return mean NLLH across batch
    return -lh.mean()


class DistNet(nn.Module):
    def __init__(self, n_input_features):
        super(DistNet, self).__init__()
        # Two hidden layers with BatchNorm + tanh
        self.fc1 = nn.Linear(n_input_features, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 16)
        self.bn2 = nn.BatchNorm1d(16)
        # Output layer: predicts [s, scale]
        self.out = nn.Linear(16, 2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        x = self.out(x)
        # ensure positive outputs via exponential
        return torch.exp(x)


class DistNetModel:
    def __init__(self, n_input_features, n_epochs=1000, wc_time_limit=3600,
                 X_valid=None, y_valid=None):  # Added validation parameters
        """
        :param n_input_features: number of instance features
        :param n_epochs: max epochs
        :param wc_time_limit: time limit in seconds
        :param X_valid: numpy array or tensor [n_val_samples, n_features]
        :param y_valid: numpy array or tensor [n_val_samples, 1]
        """
        # enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True

        # choose GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.n_epochs = n_epochs
        self.wc_time_limit = wc_time_limit

        # store validation data if provided
        if X_valid is not None and y_valid is not None:
            # Added: move validation data to device once at init
            self.X_valid = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
            self.y_valid = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
        else:
            self.X_valid = None  # No validation
            self.y_valid = None

        # model
        self.model = DistNet(n_input_features).to(self.device)
        self.batch_size = 16  # default batch size, can be overridden
        # optimizer: SGD with momentum and L2 (weight_decay)
        initial_lr = 1e-3
        final_lr = 1e-5
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=initial_lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        # exponential lr scheduler
        gamma = (final_lr / initial_lr) ** (1.0 / float(n_epochs))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def train(self, X_train, y_train):
        """
        Trains the network without using DataLoader.
        :param X_train: numpy array or torch tensor [n_samples, n_features]
        :param y_train: numpy array or torch tensor [n_samples, 1]
        """
        # convert to tensors
        X = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        n_samples = X.size(0)
        batch_size = self.batch_size

        start_time = time.time()
        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            # shuffle indices for mini-batches
            indices = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, batch_size):
                idx = indices[start:start + batch_size]
                batch_x = X[idx]
                batch_y = y[idx]

                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = nllh_loss_torch(batch_y, preds)
                loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)

            # update LR
            self.scheduler.step()
            avg_loss = epoch_loss / n_samples

            # Added: compute validation loss if data provided
            if self.X_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(self.X_valid)
                    val_loss = nllh_loss_torch(self.y_valid, val_preds).item()
                print(f"Epoch {epoch}/{self.n_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f} | Elapsed: {time.time() - start_time:.1f}s")
            else:
                print(f"Epoch {epoch}/{self.n_epochs} | Loss: {avg_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f} | Elapsed: {time.time() - start_time:.1f}s")

            # stop if exceeding time limit
            if time.time() - start_time > self.wc_time_limit:
                print(f"Time limit reached: {time.time() - start_time:.1f}s, stopping training.")
                break

        print("Training complete.")

    def predict(self, X):
        """
        Predicts [s, scale] for new instances.
        :param X: numpy array or torch tensor [n_samples, n_features]
        :return: numpy array [n_samples, 2]
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            params = self.model(X_tensor)
        return params.cpu().numpy()


if __name__ == '__main__':
    import numpy as np
    # example data
    X = np.random.rand(100, 10)
    y = np.random.lognormal(mean=np.log(1.5), sigma=0.5, size=(100, 1))
    # Added: sample validation data
    X_val = np.random.rand(20, 10)
    y_val = np.random.lognormal(mean=np.log(1.5), sigma=0.5, size=(20, 1))

    print("Sample data shapes:", X.shape, y.shape, X_val.shape, y_val.shape)
    # init model with validation
    model = DistNetModel(n_input_features=10, n_epochs=1000, wc_time_limit=3600,
                         X_valid=X_val, y_valid=y_val)
    model.train(X, y)
    print("Sample predictions:", model.predict(X[:5]))
    