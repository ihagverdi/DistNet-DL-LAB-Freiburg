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
    y_true = y_true.to(y_pred.device)
    s = y_pred[:, 0:1]
    scale = y_pred[:, 1:2]
    log_scale = torch.log(scale)
    log_true = torch.log(y_true)
    help1 = 0.5 * ((log_true - log_scale) / s) ** 2
    lh = -torch.log(s) - log_true - help1
    return -lh.mean()


class DistNet(nn.Module):
    def __init__(self, n_input_features):
        super(DistNet, self).__init__()
        self.fc1 = nn.Linear(n_input_features, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.out = nn.Linear(16, 2)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.bn1(self.fc1(x)))
        x = self.tanh(self.bn2(self.fc2(x)))
        return torch.exp(self.out(x))


class DistNetModel:
    def __init__(
        self,
        n_input_features,
        n_epochs=1000,
        wc_time_limit=3600,
        X_valid=None,
        y_valid=None,
        early_stop_patience=50
    ):
        """
        :param early_stop_patience: epochs with no improvement before stopping
        """
        torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.n_epochs = n_epochs
        self.wc_time_limit = wc_time_limit
        self.patience = early_stop_patience

        # validation data
        if X_valid is not None and y_valid is not None:
            self.X_valid = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
            self.y_valid = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
        else:
            self.X_valid = self.y_valid = None

        # model, optimizer, scheduler
        self.model = DistNet(n_input_features).to(self.device)
        initial_lr = 1e-3
        final_lr = 1e-5
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=initial_lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        gamma = (final_lr / initial_lr) ** (1.0 / float(n_epochs))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def train(self, X_train, y_train):
        X = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        n_samples = X.size(0)
        batch_size = self.batch_size if hasattr(self, 'batch_size') else 16

        best_val = float('inf')
        no_improve = 0
        start_time = time.time()

        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            indices = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, batch_size):
                idx = indices[start:start + batch_size]
                bx, by = X[idx], y[idx]
                self.optimizer.zero_grad()
                preds = self.model(bx)
                loss = nllh_loss_torch(by, preds)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * bx.size(0)

            self.scheduler.step()
            avg_train = epoch_loss / n_samples
            elapsed = time.time() - start_time

            # Validation and early stopping check
            if self.X_valid is not None:
                self.model.eval()
                with torch.no_grad():
                    vpred = self.model(self.X_valid)
                    val_loss = nllh_loss_torch(self.y_valid, vpred).item()
                print(f"Epoch {epoch}/{self.n_epochs} | Train {avg_train:.4f} | Val {val_loss:.4f} | LR {self.scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

                # Early stopping logic
                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    no_improve = 0
                    torch.save(self.model.state_dict(), 'best_model_checkpoint.pt')
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    print(f"No improvement for {self.patience} epochs, stopping early.")
                    break
            else:
                print(f"Epoch {epoch}/{self.n_epochs} | Loss {avg_train:.4f} | LR {self.scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s")

            if elapsed > self.wc_time_limit:
                print(f"Time limit reached ({elapsed:.1f}s), stopping training.")
                break

        print("Training complete.")

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()

