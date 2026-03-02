"""
ResNet-based neural network models for 4-player Doudizhu DMC training.
Architecture adapted from https://github.com/Vincentzyx/Douzero_Resnet

Four play models: landlord, landlord_next, landlord_across, landlord_prev.
One optional BidModel for future bidding phase support.

z input shape : (batch, 20, 52)  — 20 individual moves × 52-dim card encoding
               Conv1d treats dim-0 as channels (20) and dim-1 as length (52).

ResNet pipeline (z):
  conv1  : Conv1d(20 → 40, k=3, s=2, p=1)  → (batch, 40, 26)
  layer1 : BasicBlock(40  → 40,  stride=2)  → (batch,  40, 13)
  layer2 : BasicBlock(40  → 80,  stride=2)  → (batch,  80,  7)
  layer3 : BasicBlock(80  → 160, stride=2)  → (batch, 160,  4)
  flatten                                    → (batch, 640)

x input (x_batch = x_no_action + action card encoding):
  Landlord : 366 + 52 = 418 dims
  Farmers  : 370 + 52 = 422 dims

FC head (concat z-feat with x):
  Landlord : Linear(640+418=1058, 512) → 512 → 512 → 512 → 1
  Farmers  : Linear(640+422=1062, 512) → 512 → 512 → 512 → 1
"""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """1-D ResNet basic block (two Conv1d layers with a residual shortcut)."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)

        # Shortcut: identity when dims match, 1×1 Conv otherwise
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


# ---------------------------------------------------------------------------
# Shared ResNet feature extractor for z
# ---------------------------------------------------------------------------

class _ZEncoder(nn.Module):
    """
    Shared ResNet encoder for the card-history tensor z.
    Input : (batch, 20, 52)
    Output: (batch, 640)
    """
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv1d(20, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1    = nn.BatchNorm1d(40)
        self.layer1 = BasicBlock(40,  40,  stride=2)
        self.layer2 = BasicBlock(40,  80,  stride=2)
        self.layer3 = BasicBlock(80,  160, stride=2)
        # After all strides: length = 52 → 26 → 13 → 7 → 4   ⟹ 160×4 = 640

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, 20, 52)
        out = F.leaky_relu(self.bn1(self.conv1(z)))  # (batch, 40, 26)
        out = self.layer1(out)                        # (batch, 40, 13)
        out = self.layer2(out)                        # (batch, 80,  7)
        out = self.layer3(out)                        # (batch, 160, 4)
        return out.flatten(1)                         # (batch, 640)


# ---------------------------------------------------------------------------
# Play models
# ---------------------------------------------------------------------------

class LandlordResNetModel(nn.Module):
    """
    ResNet model for the Landlord position.

    z input : (batch, 20, 52)  — card-history encoding
    x input : (batch, 418)     — x_no_action (366) + action card encoding (52)
    output  : (batch, 1)       — estimated value
    """
    _Z_FEAT  = 640   # flattened ResNet output
    _X_DIM   = 418   # landlord x_batch width

    def __init__(self):
        super().__init__()
        self.z_encoder = _ZEncoder()
        combined = self._Z_FEAT + self._X_DIM   # 1058
        self.fc1 = nn.Linear(combined, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                return_value: bool = False, flags=None) -> dict:
        z_feat = self.z_encoder(z)               # (batch, 640)
        feat   = torch.cat([z_feat, x], dim=-1)  # (batch, 1058)
        feat   = F.leaky_relu(self.fc1(feat))
        feat   = F.leaky_relu(self.fc2(feat))
        feat   = F.leaky_relu(self.fc3(feat))
        values = self.fc4(feat)                  # (batch, 1)
        if return_value:
            return dict(values=values)
        if flags is not None and flags.exp_epsilon > 0 \
                and np.random.rand() < flags.exp_epsilon:
            action = torch.randint(values.shape[0], (1,))[0]
        else:
            action = torch.argmax(values, dim=0)[0]
        return dict(action=action)


class FarmerResNetModel(nn.Module):
    """
    ResNet model for all three Farmer positions
    (landlord_next, landlord_across, landlord_prev).

    z input : (batch, 20, 52)  — card-history encoding
    x input : (batch, 422)     — x_no_action (370) + action card encoding (52)
    output  : (batch, 1)       — estimated value
    """
    _Z_FEAT  = 640   # flattened ResNet output
    _X_DIM   = 422   # farmer x_batch width

    def __init__(self):
        super().__init__()
        self.z_encoder = _ZEncoder()
        combined = self._Z_FEAT + self._X_DIM   # 1062
        self.fc1 = nn.Linear(combined, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                return_value: bool = False, flags=None) -> dict:
        z_feat = self.z_encoder(z)               # (batch, 640)
        feat   = torch.cat([z_feat, x], dim=-1)  # (batch, 1062)
        feat   = F.leaky_relu(self.fc1(feat))
        feat   = F.leaky_relu(self.fc2(feat))
        feat   = F.leaky_relu(self.fc3(feat))
        values = self.fc4(feat)                  # (batch, 1)
        if return_value:
            return dict(values=values)
        if flags is not None and flags.exp_epsilon > 0 \
                and np.random.rand() < flags.exp_epsilon:
            action = torch.randint(values.shape[0], (1,))[0]
        else:
            action = torch.argmax(values, dim=0)[0]
        return dict(action=action)


# ---------------------------------------------------------------------------
# BidModel — standalone bidding evaluator (for future use)
# ---------------------------------------------------------------------------

class BidModel(nn.Module):
    """
    Model for evaluating whether to bid for Landlord.
    Architecture matches Vincentzyx/Douzero_Resnet BidModel.py.

    Default input: 60-dimensional one-hot hand encoding
      (4 suits × 15 ranks, flattened from a (4, 15) array)
    Output: scalar win-rate estimate (no sigmoid; use raw logit or apply
            sigmoid externally for a probability).

    Can be instantiated with a custom input_dim for different feature sets.
    """
    def __init__(self, input_dim: int = 60):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 1)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(self.dropout1(self.fc2(x)))
        x = torch.relu(self.dropout3(self.fc3(x)))
        x = torch.relu(self.dropout5(self.fc4(x)))
        x = torch.relu(self.dropout5(self.fc5(x)))
        return self.fc6(x)


# ---------------------------------------------------------------------------
# model_dict — used by deep_agent for evaluation
# ---------------------------------------------------------------------------

model_dict = {
    'landlord':        LandlordResNetModel,
    'landlord_next':   FarmerResNetModel,
    'landlord_across': FarmerResNetModel,
    'landlord_prev':   FarmerResNetModel,
}


# ---------------------------------------------------------------------------
# Model wrapper — used by dmc training loop
# ---------------------------------------------------------------------------

class Model:
    """
    Wrapper for the four 4-player Doudizhu ResNet play models.
    Provides a uniform interface for training (dmc.py) and actors (utils.py).
    """
    def __init__(self, device=0):
        self.models: dict = {}
        dev = torch.device('cpu') if device == 'cpu' \
              else torch.device('cuda:' + str(device))
        self.models['landlord']        = LandlordResNetModel().to(dev)
        self.models['landlord_next']   = FarmerResNetModel().to(dev)
        self.models['landlord_across'] = FarmerResNetModel().to(dev)
        self.models['landlord_prev']   = FarmerResNetModel().to(dev)

    def forward(self, position: str, z: torch.Tensor, x: torch.Tensor,
                training: bool = False, flags=None) -> dict:
        return self.models[position].forward(z, x, training, flags)

    def share_memory(self):
        for m in self.models.values():
            m.share_memory()

    def eval(self):
        for m in self.models.values():
            m.eval()

    def parameters(self, position: str):
        return self.models[position].parameters()

    def get_model(self, position: str) -> nn.Module:
        return self.models[position]

    def get_models(self) -> dict:
        return self.models
