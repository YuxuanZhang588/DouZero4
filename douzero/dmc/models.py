"""
Neural network models for 4-player Doudizhu DMC training.
Supports two z-encoder backends (selectable via --z_encoder):

  resnet      : 1-D Conv ResNet over 20-token card history (default)
  transformer : Multi-head self-attention Transformer encoder

──────────────────────────────────────────────────────────────────────
ResNet pipeline (z):
  conv1  : Conv1d(20 → 40, k=3, s=2, p=1)  → (batch, 40, 26)
  layer1 : BasicBlock(40  → 40,  stride=2)  → (batch,  40, 13)
  layer2 : BasicBlock(40  → 80,  stride=2)  → (batch,  80,  7)
  layer3 : BasicBlock(80  → 160, stride=2)  → (batch, 160,  4)
  flatten                                    → (batch, 640)

Transformer pipeline (z):
  proj   : Linear(52, 128)                  → (batch, 20, 128)
  + learned positional embedding (20, 128)
  encoder: TransformerEncoder(d=128, heads=4, layers=4, ff=512)
  mean-pool over 20 tokens                  → (batch, 128)
  out_proj: Linear(128, 640)                → (batch, 640)

Both encoders output (batch, 640) — identical downstream FC head.
──────────────────────────────────────────────────────────────────────
z input shape : (batch, 20, 52)  — 20 individual moves × 52-dim card encoding
x input (x_batch = x_no_action + action card encoding):
  Landlord : 366 + 52 = 418 dims
  Farmers  : 370 + 52 = 422 dims

FC head (Dueling, concat z-feat with x):
  Landlord : Linear(640+418=1058, 512) → 512 → 512 → 512 → 1  (advantage)
             Linear(640→256→1)                                  (value)
  Farmers  : Linear(640+422=1062, 512) → 512 → 512 → 512 → 1
             Linear(640→256→1)
"""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """1-D ResNet basic block (two Conv1d layers with a residual shortcut).

    Uses affine=False BatchNorm to prevent the learnable gamma parameter from
    collapsing to zero during training (a known issue with BN in residual nets).
    Fixed gamma=1 / beta=0 keeps normalization benefits without the collapse.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=True)
        self.bn1   = nn.BatchNorm1d(planes, affine=False)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.bn2   = nn.BatchNorm1d(planes, affine=False)

        # Shortcut: identity when dims match, 1×1 Conv otherwise
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm1d(self.expansion * planes, affine=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


# ---------------------------------------------------------------------------
# z-encoder backends
# ---------------------------------------------------------------------------

class _ZEncoder(nn.Module):
    """
    ResNet encoder for the card-history tensor z.
    Input : (batch, 20, 52)
    Output: (batch, 640)
    """
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv1d(20, 40, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1    = nn.BatchNorm1d(40, affine=False)
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


class _ZEncoderTransformer(nn.Module):
    """
    Transformer encoder for the card-history tensor z.

    Each of the 20 tokens represents one historical card play (52-dim).
    Self-attention lets every token attend to every other token, capturing
    long-range dependencies that Conv1d with limited receptive field cannot.

    Architecture:
      proj     : Linear(52, d_model)            token projection
      pos_emb  : Embedding(20, d_model)         learned turn-order embedding
      player_emb: Embedding(4, d_model)         learned player-ID embedding
                  (turn i belongs to player i%4 relative to current player)
      encoder  : TransformerEncoder(d_model, nhead, num_layers, dim_feedforward)
      mean-pool over sequence → Linear(d_model, 640)

    Input : (batch, 20, 52)
    Output: (batch, 640)
    """
    N_TOKENS  = 20
    CARD_DIM  = 52
    N_PLAYERS = 4

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Token projection
        self.proj      = nn.Linear(self.CARD_DIM, d_model)
        # Positional + player-ID embeddings (both learned)
        self.pos_emb    = nn.Embedding(self.N_TOKENS, d_model)
        self.player_emb = nn.Embedding(self.N_PLAYERS, d_model)
        # Register position indices and player indices as buffers
        self.register_buffer('_pos_ids',
            torch.arange(self.N_TOKENS))              # (20,)
        self.register_buffer('_player_ids',
            torch.arange(self.N_TOKENS) % self.N_PLAYERS)  # (20,)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,          # Pre-LN (more stable than post-LN)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output projection to match ResNet's 640-dim interface
        self.out_proj = nn.Linear(d_model, 640)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, 20, 52)
        x = self.proj(z)                                    # (batch, 20, d_model)
        x = x + self.pos_emb(self._pos_ids)                 # broadcast over batch
        x = x + self.player_emb(self._player_ids)           # relative player ID
        x = self.encoder(x)                                 # (batch, 20, d_model)
        x = x.mean(dim=1)                                   # (batch, d_model)
        return self.out_proj(x)                             # (batch, 640)


def _build_z_encoder(z_encoder: str = 'resnet') -> nn.Module:
    """Factory: return the appropriate z-encoder by name."""
    if z_encoder == 'transformer':
        return _ZEncoderTransformer()
    return _ZEncoder()


# ---------------------------------------------------------------------------
# Play models
# ---------------------------------------------------------------------------

class LandlordResNetModel(nn.Module):
    """
    Landlord model — supports both ResNet and Transformer z-encoders.

    z input : (batch, 20, 52)  — card-history encoding
    x input : (batch, 418)     — x_no_action (366) + action card encoding (52)
    output  : (batch, 1)       — estimated Q-value (Dueling: V(s) + A(s,a) - mean_A)
    """
    _Z_FEAT  = 640   # z-encoder output (both backends output 640)
    _X_DIM   = 418   # landlord x_batch width

    def __init__(self, z_encoder: str = 'resnet'):
        super().__init__()
        self.z_encoder = _build_z_encoder(z_encoder)
        combined = self._Z_FEAT + self._X_DIM   # 1058
        # Shared advantage trunk
        self.fc1  = nn.Linear(combined, 512)
        self.ln1  = nn.LayerNorm(512)
        self.fc2  = nn.Linear(512, 512)
        self.ln2  = nn.LayerNorm(512)
        self.fc3  = nn.Linear(512, 512)
        self.ln3  = nn.LayerNorm(512)
        self.fc4  = nn.Linear(512, 1)              # advantage head (name kept for compat)
        # Dueling value branch (state-only, no action info)
        self.fc_v1 = nn.Linear(self._Z_FEAT, 256)
        self.fc_v2 = nn.Linear(256, 1)

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                return_value: bool = False, flags=None,
                exp_epsilon: float = None) -> dict:
        z_feat = self.z_encoder(z)                            # (batch, 640)
        # Advantage stream: concat z_feat with action features
        feat   = torch.cat([z_feat, x], dim=-1)               # (batch, 1058)
        feat   = F.leaky_relu(self.ln1(self.fc1(feat)))       # (batch, 512)
        feat   = F.leaky_relu(self.ln2(self.fc2(feat)))       # (batch, 512)
        feat   = F.leaky_relu(self.ln3(self.fc3(feat)))       # (batch, 512)
        adv    = self.fc4(feat)                               # (batch, 1)
        # Value stream: state-only (z_feat, same for all candidate actions)
        val    = F.leaky_relu(self.fc_v1(z_feat))             # (batch, 256)
        val    = self.fc_v2(val)                              # (batch, 1)
        # Dueling combination: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
        values = val + adv - adv.mean(dim=0, keepdim=True)    # (batch, 1)
        if return_value:
            return dict(values=values)
        eps = exp_epsilon if exp_epsilon is not None else (
              flags.exp_epsilon if flags is not None else 0.0)
        if eps > 0 and np.random.rand() < eps:
            action = torch.randint(values.shape[0], (1,))[0]
        else:
            action = torch.argmax(values, dim=0)[0]
        return dict(action=action)


class FarmerResNetModel(nn.Module):
    """
    Farmer model — supports both ResNet and Transformer z-encoders.
    Covers landlord_next, landlord_across, landlord_prev.

    z input : (batch, 20, 52)  — card-history encoding
    x input : (batch, 422)     — x_no_action (370) + action card encoding (52)
    output  : (batch, 1)       — estimated Q-value (Dueling: V(s) + A(s,a) - mean_A)
    """
    _Z_FEAT  = 640   # z-encoder output (both backends output 640)
    _X_DIM   = 422   # farmer x_batch width

    def __init__(self, z_encoder: str = 'resnet'):
        super().__init__()
        self.z_encoder = _build_z_encoder(z_encoder)
        combined = self._Z_FEAT + self._X_DIM   # 1062
        # Shared advantage trunk
        self.fc1  = nn.Linear(combined, 512)
        self.ln1  = nn.LayerNorm(512)
        self.fc2  = nn.Linear(512, 512)
        self.ln2  = nn.LayerNorm(512)
        self.fc3  = nn.Linear(512, 512)
        self.ln3  = nn.LayerNorm(512)
        self.fc4  = nn.Linear(512, 1)              # advantage head (name kept for compat)
        # Dueling value branch (state-only, no action info)
        self.fc_v1 = nn.Linear(self._Z_FEAT, 256)
        self.fc_v2 = nn.Linear(256, 1)

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                return_value: bool = False, flags=None,
                exp_epsilon: float = None) -> dict:
        z_feat = self.z_encoder(z)                            # (batch, 640)
        # Advantage stream: concat z_feat with action features
        feat   = torch.cat([z_feat, x], dim=-1)               # (batch, 1062)
        feat   = F.leaky_relu(self.ln1(self.fc1(feat)))       # (batch, 512)
        feat   = F.leaky_relu(self.ln2(self.fc2(feat)))       # (batch, 512)
        feat   = F.leaky_relu(self.ln3(self.fc3(feat)))       # (batch, 512)
        adv    = self.fc4(feat)                               # (batch, 1)
        # Value stream: state-only (z_feat, same for all candidate actions)
        val    = F.leaky_relu(self.fc_v1(z_feat))             # (batch, 256)
        val    = self.fc_v2(val)                              # (batch, 1)
        # Dueling combination: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
        values = val + adv - adv.mean(dim=0, keepdim=True)    # (batch, 1)
        if return_value:
            return dict(values=values)
        eps = exp_epsilon if exp_epsilon is not None else (
              flags.exp_epsilon if flags is not None else 0.0)
        if eps > 0 and np.random.rand() < eps:
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
# Legacy LSTM models (used by old checkpoints trained before ResNet migration)
# z input shape: (batch, 5, 208)  — 5 rounds × 4 players × 52 cards (LSTM)
# ---------------------------------------------------------------------------

class LegacyLandlordModel(nn.Module):
    """Original LSTM-based Landlord model. x_batch = 366+52 = 418 dims."""
    def __init__(self):
        super().__init__()
        self.lstm   = nn.LSTM(208, 128, batch_first=True)
        self.dense1 = nn.Linear(546, 512)  # 128 + 418
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, _ = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        x = torch.relu(self.dense5(x))
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        action = torch.argmax(x, dim=0)[0]
        return dict(action=action)


class LegacyFarmerModel(nn.Module):
    """Original LSTM-based Farmer model. x_batch = 370+52 = 422 dims."""
    def __init__(self):
        super().__init__()
        self.lstm   = nn.LSTM(208, 128, batch_first=True)
        self.dense1 = nn.Linear(550, 512)  # 128 + 422
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, _ = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        x = torch.relu(self.dense5(x))
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        action = torch.argmax(x, dim=0)[0]
        return dict(action=action)


legacy_model_dict = {
    'landlord':        LegacyLandlordModel,
    'landlord_next':   LegacyFarmerModel,
    'landlord_across': LegacyFarmerModel,
    'landlord_prev':   LegacyFarmerModel,
}


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
    Wrapper for the four 4-player Doudizhu play models.
    Provides a uniform interface for training (dmc.py) and actors (utils.py).

    z_encoder : 'resnet' (default) or 'transformer'
    """
    def __init__(self, device=0, z_encoder: str = 'resnet'):
        self.models: dict = {}
        dev = torch.device('cpu') if device == 'cpu' \
              else torch.device('cuda:' + str(device))
        self.models['landlord']        = LandlordResNetModel(z_encoder).to(dev)
        self.models['landlord_next']   = FarmerResNetModel(z_encoder).to(dev)
        self.models['landlord_across'] = FarmerResNetModel(z_encoder).to(dev)
        self.models['landlord_prev']   = FarmerResNetModel(z_encoder).to(dev)

    def forward(self, position: str, z: torch.Tensor, x: torch.Tensor,
                training: bool = False, flags=None,
                exp_epsilon: float = None) -> dict:
        return self.models[position].forward(z, x, training, flags, exp_epsilon)

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
