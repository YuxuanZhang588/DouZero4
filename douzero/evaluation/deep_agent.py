import torch
import numpy as np

from douzero.env.env import get_obs


def _load_model(position, model_path, legacy=False):
    if legacy:
        from douzero.dmc.models import legacy_model_dict
        model = legacy_model_dict[position]()
    else:
        from douzero.dmc.models import model_dict
        model = model_dict[position]()
    model_state_dict = model.state_dict()
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if use_cuda:
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if use_cuda:
        model.cuda()
    model.eval()
    return model


class DeepAgent:
    """
    Deep model agent for evaluation.
    Set legacy=True to load old LSTM checkpoints (z will be reshaped
    from the current (B,20,52) env format to the old (B,5,208) LSTM format).
    """
    def __init__(self, position, model_path, legacy=False):
        self.model  = _load_model(position, model_path, legacy=legacy)
        self.legacy = legacy

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset)

        z_batch = torch.from_numpy(obs['z_batch']).float()  # (B, 20, 52)
        x_batch = torch.from_numpy(obs['x_batch']).float()

        if self.legacy:
            # Reshape from (B, 20, 52) → (B, 5, 208) for old LSTM models
            B = z_batch.shape[0]
            z_batch = z_batch.reshape(B, 5, 208)

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()

        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        return best_action
