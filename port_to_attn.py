"""
Port resnet_v3 checkpoint to attn_v1 (Transformer z-encoder).
FC layers (fc1-fc4, fc_v1, fc_v2, ln1-ln3) carry over unchanged.
z_encoder (transformer) starts from fresh random init.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from douzero.dmc.models import LandlordResNetModel, FarmerResNetModel, Model
from douzero.dmc.utils import POSITIONS

SRC = os.path.join('douzero_checkpoints', 'resnet_v3', 'model.tar')
DST_DIR = os.path.join('douzero_checkpoints', 'attn_v1')
DST = os.path.join(DST_DIR, 'model.tar')
os.makedirs(DST_DIR, exist_ok=True)

print(f'Loading {SRC} ...')
old = torch.load(SRC, map_location='cpu')
old_states = old['model_state_dict']

model_classes = {
    'landlord':        LandlordResNetModel,
    'landlord_next':   FarmerResNetModel,
    'landlord_across': FarmerResNetModel,
    'landlord_prev':   FarmerResNetModel,
}
new_states = {}
for pos, cls in model_classes.items():
    m = cls(z_encoder='transformer')
    missing, unexpected = m.load_state_dict(old_states[pos], strict=False)
    z_enc_missing  = [k for k in missing   if k.startswith('z_encoder')]
    fc_missing     = [k for k in missing   if not k.startswith('z_encoder')]
    z_enc_unexpect = [k for k in unexpected if k.startswith('z_encoder')]
    print(f'  [{pos}] z_encoder keys fresh (transformer init): {len(z_enc_missing)}')
    if fc_missing:
        print(f'  [{pos}] WARNING other missing keys: {fc_missing}')
    if z_enc_unexpect:
        print(f'  [{pos}] old z_encoder keys dropped: {len(z_enc_unexpect)}')
    new_states[pos] = m.state_dict()

torch.save({
    'model_state_dict': new_states,
    'optimizer_state_dict': {p: {} for p in POSITIONS},
    'stats': {},
    'flags': {},
    'frames': 0,
    'position_frames': {p: 0 for p in POSITIONS},
}, DST)
print(f'\nSaved to {DST}')

# Smoke test
print('\nSmoke-testing Transformer forward pass ...')
ll = LandlordResNetModel(z_encoder='transformer')
ff = FarmerResNetModel(z_encoder='transformer')

z   = torch.zeros(5, 32, 52)   # 32-token history
x_l = torch.zeros(5, 418)
x_f = torch.zeros(5, 422)

out_l = ll.forward(z, x_l, return_value=True)
out_f = ff.forward(z, x_f, return_value=True)
print(f'  Landlord values shape : {out_l["values"].shape}')
print(f'  Farmer   values shape : {out_f["values"].shape}')

wrapper = Model(device='cpu', z_encoder='transformer')
out = wrapper.forward('landlord', z, x_l, exp_epsilon=1.0)
print(f'  Model.forward action  : {out["action"]}')

# Also verify resnet still works with 32-token input
ll_r = LandlordResNetModel(z_encoder='resnet')
out_r = ll_r.forward(z, x_l, return_value=True)
print(f'  ResNet values shape   : {out_r["values"].shape}  (backward compat)')

print('\nAll checks passed.')
