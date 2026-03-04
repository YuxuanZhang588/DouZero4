"""
One-shot script: port resnet_v2 model.tar weights into the new resnet_v3
architecture (Dueling + LayerNorm), then do a quick smoke-test forward pass.

New layers (ln1-ln3, fc_v1, fc_v2) are not in the old checkpoint, so they
start from random init — which is intentional; the ResNet z-encoder and
fc1-fc4 advantage trunk weights carry over.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from douzero.dmc.models import LandlordResNetModel, FarmerResNetModel, Model
from douzero.dmc.utils import POSITIONS

SRC = os.path.join('douzero_checkpoints', 'resnet_v2', 'model.tar')
DST_DIR = os.path.join('douzero_checkpoints', 'resnet_v3')
DST = os.path.join(DST_DIR, 'model.tar')

os.makedirs(DST_DIR, exist_ok=True)

print(f'Loading {SRC} ...')
old = torch.load(SRC, map_location='cpu')
old_states = old['model_state_dict']

# Build fresh v3 models
model_classes = {
    'landlord':        LandlordResNetModel,
    'landlord_next':   FarmerResNetModel,
    'landlord_across': FarmerResNetModel,
    'landlord_prev':   FarmerResNetModel,
}
new_states = {}
for pos, cls in model_classes.items():
    m = cls()
    missing, unexpected = m.load_state_dict(old_states[pos], strict=False)
    print(f'  [{pos}] missing (new, random-init): {missing}')
    if unexpected:
        print(f'  [{pos}] unexpected (ignored):       {unexpected}')
    new_states[pos] = m.state_dict()

# Save v3 checkpoint — reset frames/stats so training starts clean
torch.save({
    'model_state_dict': new_states,
    'optimizer_state_dict': {p: {} for p in POSITIONS},   # fresh optimizers
    'stats': {},
    'flags': {},
    'frames': 0,
    'position_frames': {p: 0 for p in POSITIONS},
}, DST)
print(f'\nSaved port to {DST}')

# ----- Quick smoke test -----
print('\nSmoke-testing forward pass ...')
device = torch.device('cpu')
landlord = LandlordResNetModel().to(device)
farmer   = FarmerResNetModel().to(device)

z = torch.zeros(3, 20, 52)     # batch=3 candidate actions
x_l = torch.zeros(3, 418)
x_f = torch.zeros(3, 422)

out_l = landlord.forward(z, x_l, return_value=True)
out_f = farmer.forward(z, x_f, return_value=True)
print(f'  Landlord values shape: {out_l["values"].shape}')
print(f'  Farmer   values shape: {out_f["values"].shape}')

# Test Model wrapper with exp_epsilon override
wrapper = Model(device='cpu')
out_action = wrapper.forward('landlord', z, x_l, exp_epsilon=1.0)   # random action
print(f'  Model.forward action (random eps=1.0): {out_action["action"]}')

print('\nAll checks passed. Ready to train resnet_v3.')
