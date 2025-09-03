# ==================================================================================
#  Bayesian Temporal Convolutional Network for Crop-Yield Prediction (for reference)
#  Author : wsy
# ==================================================================================

import os
from math import sqrt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------------------
# 1. Hyper-parameters & File Paths
# ------------------------------------------------------------------
DATA_FILE      = "Data/Phen_combined_xy_normalized.csv"
TEST_YEAR      = 17          # first two digits of codeID
FEATURES       = [
    "EVI", "NDVI", "Qair_f_inst", "SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst",
    "Tem_Nonzero", "evaporation_from_the_top_of_canopy_sum",
    "evaporation_from_vegetation_transpiration_sum", "potential_evaporation_sum",
    "soil_temperature_level_1", "soil_temperature_level_2",
    "surface_net_solar_radiation_sum", "surface_pressure",
    "temperature_2m_max", "temperature_2m_min", "total_evaporation_sum",
    "total_precipitation_sum", "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2"
]
N_TIMESTEPS    = 6           # temporal depth of each sample
BATCH_SIZE     = 64
EPOCHS         = 200
LR             = 1e-3
PATIENCE       = 5
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 2. Data Loading & Pre-processing
# ------------------------------------------------------------------
def load_data(path: str):
    """Load CSV and return train/test tensors ready for DataLoader."""
    df = pd.read_csv(path)

    X_train, X_test, y_train, y_test, code_test = [], [], [], [], []

    for _, g in df.groupby("codeID"):
        feat = g[FEATURES].values.reshape(N_TIMESTEPS, -1)        # (T, C)
        targ = g["kg_ha"].iloc[0]                                 # scalar

        if int(str(g["codeID"].iloc[0])[:2]) == TEST_YEAR:
            X_test.append(feat)
            y_test.append(targ)
            code_test.append(g["codeID"].iloc[0])
        else:
            X_train.append(feat)
            y_train.append(targ)

    # Convert to ndarray
    X_train = np.array(X_train, dtype=np.float32)                # (N, T, C)
    X_test  = np.array(X_test,  dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    y_test  = np.array(y_test,  dtype=np.float32).reshape(-1, 1)

    # Standardise features (fit on train, apply on test)
    scaler = StandardScaler()
    N, T, C = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, C)).reshape(N, T, C)
    X_test  = scaler.transform(X_test.reshape(-1, C)).reshape(-1, T, C)

    # Log-transform target to stabilise variance
    y_train_log = np.log1p(y_train)
    y_test_log  = np.log1p(y_test)

    # Convert to torch tensor  (batch, channels, length)
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_test  = torch.tensor(X_test,  dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.tensor(y_train_log, dtype=torch.float32)
    y_test  = torch.tensor(y_test_log,  dtype=torch.float32)

    return X_train, X_test, y_train, y_test, code_test

X_train, X_test, y_train, y_test, code_test = load_data(DATA_FILE)

train_set = TensorDataset(X_train, y_train)
test_set  = TensorDataset(X_test,  y_test)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------------
# 3. Model Architecture
# ------------------------------------------------------------------
class TemporalBlock(nn.Module):
    """Single 1-D causal convolution block with ReLU + Dropout.(just for reference)"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.relu(self.conv(x)))

class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation."""
    def __init__(self, in_ch: int, channels: list, kernel: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch if i == 0 else channels[i-1],
                                        out_ch, kernel, dilation=2**i, dropout=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BayesianTCN(nn.Module):
    """Bayesian output layer on top of TCN: predicts mean + sigma."""
    def __init__(self, in_ch: int, channels: list, kernel: int, dropout: float = 0.0):
        super().__init__()
        self.tcn   = TemporalConvNet(in_ch, channels, kernel, dropout)
        self.mean  = nn.Linear(channels[-1], 1)
        self.sigma = nn.Linear(channels[-1], 1)

    def forward(self, x):
        # x: (B, C, T)  ->  (B, hidden, T)
        feat = self.tcn(x)[:, :, -1]          # last time-step
        mu   = self.mean(feat).squeeze(-1)
        sig  = torch.exp(self.sigma(feat)).squeeze(-1).clamp(min=1e-4)
        return mu, sig

    def loss_fn(self, pred, y):
        mu, sig = pred
        return -torch.distributions.Normal(mu, sig).log_prob(y).mean()

# ------------------------------------------------------------------
# 4. Early Stopping Utility
# ------------------------------------------------------------------
class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose  = verbose
        self.counter  = 0
        self.best     = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# ------------------------------------------------------------------
# 5. Training Loop
# ------------------------------------------------------------------
model = BayesianTCN(in_ch=len(FEATURES), channels=[64, 32, 64], kernel=3, dropout=0.2).to(DEVICE)
optimiser = optim.Adam(model.parameters(), lr=LR)

def train_model(net, opt, loader, epochs, patience=None):
    net.train()
    early_stop = EarlyStopping(patience, verbose=True) if patience else None
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = net.loss_fn(net(xb), yb.squeeze())
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        print(f"Epoch {epoch:03d} | loss = {avg:.4f}")
        if early_stop:
            early_stop(avg)
            if early_stop.early_stop:
                print("Early stopping triggered")
                break

train_model(model, optimiser, train_loader, EPOCHS, patience=PATIENCE)

# ------------------------------------------------------------------
# 6. Evaluation & Uncertainty Estimation
# ------------------------------------------------------------------
@torch.no_grad()
def predict(net, loader, mc=100):
    """Return MC predictions (mean & std) in original scale."""
    net.eval()
    means, stds = [], []
    for _ in range(mc):
        preds = []
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            mu, sig = net(xb)
            preds.append(torch.stack([mu, sig], dim=-1))
        preds = torch.cat(preds, dim=0)          # (N, 2)
        means.append(torch.exp(preds[:, 0]) - 1) # reverse log1p
        stds.append(preds[:, 1])
    means = torch.stack(means, dim=0)            # (MC, N)
    stds  = torch.stack(stds,  dim=0)
    epistemic = means.std(dim=0)                 # model uncertainty
    aleatoric = stds.mean(dim=0)                 # data noise
    return means.mean(dim=0), epistemic, aleatoric

pred_mean, epistemic, aleatoric = predict(model, test_loader, mc=100)

r2  = r2_score(np.expm1(y_test.numpy()), pred_mean.numpy())
rmse = sqrt(mean_squared_error(np.expm1(y_test.numpy()), pred_mean.numpy()))

print("--------------------------------------------------")
print("Test metrics (original kg/ha scale)")
print(f"R²   = {r2:.3f}")
print(f"RMSE = {rmse:.3f}")
print(f"Mean epistemic uncertainty : {epistemic.mean():.3f}")
print(f"Mean aleatoric uncertainty : {aleatoric.mean():.3f}")