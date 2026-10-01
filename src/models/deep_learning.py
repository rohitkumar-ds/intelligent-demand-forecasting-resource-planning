"""Compact PyTorch sequence models with fine-tuning, early stopping, and self-contained persistence."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch import nn
from .base import BaseForecastModel


class _Net(nn.Module):
    def __init__(self, hidden_dim: int = 24, num_layers: int = 1, dropout: float = 0.0, attention: bool = False):
        super().__init__()
        self.attention = attention
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True) if attention else None
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        if self.attn is not None:
            h, _ = self.attn(h, h, h)
        return self.fc(h[:, -1, :]).squeeze(-1)


class _DeepForecaster(BaseForecastModel):
    name = "Deep"

    def __init__(
        self,
        attention: bool = False,
        window: int = 28,
        hidden_dim: int = 24,
        num_layers: int = 1,
        learning_rate: float = 0.004,
        epochs: int = 30,
        batch_size: int = 64,
    ):
        super().__init__()
        self.attention = attention
        self.window = window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.net: _Net | None = None
        self.mean: float = 0.0
        self.scale: float = 1.0

    def fit(self, df: pd.DataFrame, target: str, tune: bool = False) -> _DeepForecaster:
        self.target = target
        y = df.sort_values("date")[target].to_numpy(np.float32)
        
        # Temporal train/val split (85% train, 15% validation)
        split_idx = int(len(y) * 0.85)
        train_y, val_y = y[:split_idx], y[split_idx:]
        
        self.mean = float(np.mean(train_y))
        self.scale = float(max(np.std(train_y), 1e-6))

        def create_windows(series: np.ndarray):
            z = (series - self.mean) / self.scale
            X = np.stack([z[i - self.window:i] for i in range(self.window, len(z))])
            Y = z[self.window:]
            return torch.tensor(X[:, :, None], dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

        tx_train, ty_train = create_windows(train_y)
        
        # Include a warm-up buffer for the validation sequence
        val_buffer_y = y[split_idx - self.window:]
        tx_val, ty_val = create_windows(val_buffer_y)

        best_config = {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "lr": self.learning_rate,
        }

        if tune:
            candidates = [
                {"hidden_dim": 16, "num_layers": 1, "lr": 0.005},
                {"hidden_dim": 24, "num_layers": 1, "lr": 0.004},
                {"hidden_dim": 32, "num_layers": 2, "lr": 0.003},
            ]
            best_val_loss = float("inf")

            for cfg in candidates:
                net = _Net(
                    hidden_dim=cfg["hidden_dim"],
                    num_layers=cfg["num_layers"],
                    attention=self.attention,
                )
                opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
                loss_fn = nn.MSELoss()

                # Quick trial training loop
                net.train()
                for _ in range(15):
                    opt.zero_grad()
                    loss = loss_fn(net(tx_train), ty_train)
                    loss.backward()
                    opt.step()

                net.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(net(tx_val), ty_val).item())

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = cfg

        self.hidden_dim = best_config["hidden_dim"]
        self.num_layers = best_config["num_layers"]
        self.learning_rate = best_config["lr"]

        # Final network instantiation
        self.net = _Net(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            attention=self.attention,
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_state = None
        min_val_loss = float("inf")
        patience, patience_counter = 5, 0

        for _ in range(self.epochs):
            self.net.train()
            optimizer.zero_grad()
            loss = criterion(self.net(tx_train), ty_train)
            loss.backward()
            optimizer.step()

            self.net.eval()
            with torch.no_grad():
                val_loss = float(criterion(self.net(tx_val), ty_val).item())

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        # In-sample residuals
        self.net.eval()
        with torch.no_grad():
            train_preds = self.net(tx_train).numpy() * self.scale + self.mean
            val_preds = self.net(tx_val).numpy() * self.scale + self.mean

        self.residuals = train_y[self.window:] - train_preds
        # Use validation residuals as unbiased calibration error
        self.set_calibration_residuals(val_y - val_preds)
        self.fitted = True
        return self

    def predict(
        self,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self.fitted or self.net is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        self.net.eval()
        y_hist = history.sort_values("date")[self.target].to_numpy(np.float32).tolist()

        for _ in range(horizon):
            recent_seq = np.asarray(y_hist[-self.window:], dtype=np.float32)
            norm_seq = (recent_seq - self.mean) / self.scale
            tensor_seq = torch.tensor(norm_seq[None, :, None], dtype=torch.float32)

            with torch.no_grad():
                pred_norm = self.net(tensor_seq).item()
                pred_val = float(pred_norm * self.scale + self.mean)

            y_hist.append(max(pred_val, 0.0))

        return np.asarray(y_hist[-horizon:], dtype=float)

    def save(self, path: str | Path) -> None:
        """Saves weights along with metadata so the network can be reconstructed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.name,
            "target": self.target,
            "attention": self.attention,
            "window": self.window,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "mean": self.mean,
            "scale": self.scale,
            "residuals": self.residuals,
            "calibration_residuals": self.calibration_residuals,
            "fitted": self.fitted,
            "state_dict": self.net.state_dict() if self.net is not None else None,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> _DeepForecaster:
        """Restores model structure, weights, and normalization parameters."""
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)

        model = cls()
        model.target = payload.get("target")
        model.attention = payload.get("attention", False)
        model.window = payload.get("window", 28)
        model.hidden_dim = payload.get("hidden_dim", 24)
        model.num_layers = payload.get("num_layers", 1)
        model.learning_rate = payload.get("learning_rate", 0.004)
        model.epochs = payload.get("epochs", 30)
        model.mean = payload.get("mean", 0.0)
        model.scale = payload.get("scale", 1.0)
        model.residuals = payload.get("residuals", np.array([]))
        model.calibration_residuals = payload.get("calibration_residuals", np.array([]))
        model.fitted = payload.get("fitted", False)

        state_dict = payload.get("state_dict")
        if state_dict is not None:
            model.net = _Net(
                hidden_dim=model.hidden_dim,
                num_layers=model.num_layers,
                attention=model.attention,
            )
            model.net.load_state_dict(state_dict)
            model.net.eval()

        return model


class LSTMForecaster(_DeepForecaster):
    name = "LSTM"

    def __init__(self, **kwargs):
        super().__init__(attention=False, **kwargs)


class AttentionForecaster(_DeepForecaster):
    name = "Temporal Attention"

    def __init__(self, **kwargs):
        super().__init__(attention=True, **kwargs)
