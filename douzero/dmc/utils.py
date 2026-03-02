"""
Utility functions for 4-player Doudizhu DMC training.
"""
import os 
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time

import torch 
from torch import multiprocessing as mp

from .env_utils import Environment
from douzero.env import Env
from douzero.env.env import _cards2array

# 4-player positions
POSITIONS = ['landlord', 'landlord_next', 'landlord_across', 'landlord_prev']

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_env(flags):
    return Env(flags.objective)

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_optimizers(flags, learner_model):
    """
    Create four optimizers for the four positions in 4-player Doudizhu.
    """
    optimizers = {}
    for position in POSITIONS:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers

def create_buffers(flags, device_iterator):
    """
    Create buffers for 4-player Doudizhu training.
    Each device will have four buffers for the four positions.
    
    Feature dimensions:
    - Landlord x_no_action: 366 dims  (6*52 + 3*13 + 15)
    - Farmer x_no_action: 370 dims    (6*52 + 17 + 2*13 + 15)
    - Action encoding: 52 dims (no jokers)
    - ResNet input z: 20 x 52 (20 individual moves x 52-dim card encoding)
    """
    T = flags.unroll_length
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in POSITIONS:
            x_dim = 366 if position == 'landlord' else 370
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 52), dtype=torch.int8),  # 52 cards, no jokers
                obs_z=dict(size=(T, 20, 52), dtype=torch.int8),   # 20 moves x 52-dim card encoding
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers

def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    Actor process for 4-player Doudizhu.
    Generates data from the environment and sends to buffer.
    """
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in POSITIONS}
        episode_return_buf = {p: [] for p in POSITIONS}
        target_buf = {p: [] for p in POSITIONS}
        obs_x_no_action_buf = {p: [] for p in POSITIONS}
        obs_action_buf = {p: [] for p in POSITIONS}
        obs_z_buf = {p: [] for p in POSITIONS}
        size = {p: 0 for p in POSITIONS}

        position, obs, env_output = env.initial()

        while True:
            while True:
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))
                size[position] += 1
                position, obs, env_output = env.step(action)
                if env_output['done']:
                    for p in POSITIONS:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)

                            # Landlord gets positive reward, farmers get negative
                            episode_return = env_output['episode_return'] if p == 'landlord' else -env_output['episode_return']
                            episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break

            for p in POSITIONS:
                while size[p] > T: 
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e

def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix
