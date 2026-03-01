"""
Neural network models for 4-player Doudizhu DMC training.
Four models: landlord, landlord_next, landlord_across, landlord_prev.
"""

import numpy as np

import torch
from torch import nn


class LandlordLstmModel(nn.Module):
    """
    Landlord model for 4-player Doudizhu.
    
    LSTM input: 208 (5 rounds × 4 players × 52 cards = 5 × 208)
    Dense1 input: x_no_action (364) + LSTM output (128) = 492
    
    x_no_action features (364 dims):
    - my_handcards: 52
    - other_handcards: 52
    - last_action: 52
    - farmer_next_played: 52
    - farmer_across_played: 52
    - farmer_prev_played: 52
    - farmer_next_num_cards: 13
    - farmer_across_num_cards: 13
    - farmer_prev_num_cards: 13
    - bomb_num: 15
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(208, 128, batch_first=True)
        self.dense1 = nn.Linear(546, 512)  # LSTM (128) + x_batch (366 + 52 action)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)


class FarmerLstmModel(nn.Module):
    """
    Farmer model for 4-player Doudizhu (used by all 3 farmer positions).
    
    LSTM input: 208 (5 rounds × 4 players × 52 cards = 5 × 208)
    Dense1 input: x_no_action (369) + LSTM output (128) = 497
    
    x_no_action features (369 dims):
    - my_handcards: 52
    - other_handcards: 52
    - last_action: 52
    - landlord_played: 52
    - teammate1_played: 52
    - teammate2_played: 52
    - landlord_num_cards: 17 (max 16)
    - teammate1_num_cards: 13 (max 12)
    - teammate2_num_cards: 13 (max 12)
    - bomb_num: 15
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(208, 128, batch_first=True)
        self.dense1 = nn.Linear(550, 512)  # LSTM (128) + x_batch (370 + 52 action)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)


# Model dict is used in evaluation
model_dict = {}
model_dict['landlord'] = LandlordLstmModel
model_dict['landlord_next'] = FarmerLstmModel
model_dict['landlord_across'] = FarmerLstmModel
model_dict['landlord_prev'] = FarmerLstmModel


class Model:
    """
    Wrapper for the four 4-player Doudizhu models.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['landlord'] = LandlordLstmModel().to(torch.device(device))
        self.models['landlord_next'] = FarmerLstmModel().to(torch.device(device))
        self.models['landlord_across'] = FarmerLstmModel().to(torch.device(device))
        self.models['landlord_prev'] = FarmerLstmModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_next'].share_memory()
        self.models['landlord_across'].share_memory()
        self.models['landlord_prev'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_next'].eval()
        self.models['landlord_across'].eval()
        self.models['landlord_prev'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
