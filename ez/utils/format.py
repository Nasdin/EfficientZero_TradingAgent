# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import os
import torch
import random
import logging
import numpy as np
import subprocess as sp
from ray.util.queue import Queue

class RayQueue(object):
    def __init__(self, threshold=15, size=20):
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, batch):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(batch)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def get_len(self):
        return self.queue.qsize()


class PreQueue(object):
    def __init__(self, threshold=15, size=20):
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, batch):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(batch)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def get_len(self):
        return self.queue.qsize()


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def transform_one2(x):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1) + 0.001 * x

def transform_one(x):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1) + 0.001 * x

def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

class DiscreteSupport(object):
    def __init__(self, config=None):
        if config:
            self.env = config.env.env
            if self.env in ['DMC', 'Gym']:
                assert config.model.reward_support.bins == config.model.value_support.bins
                self.size = config.model.reward_support.bins
            else:
                assert config.model.reward_support.range[0] == config.model.value_support.range[0]
                assert config.model.reward_support.range[1] == config.model.value_support.range[1]
                assert config.model.reward_support.scale == config.model.value_support.scale
                self.min = config.model.reward_support.range[0]
                self.max = config.model.reward_support.range[1]
                self.scale = config.model.reward_support.scale
                self.range = np.arange(self.min, self.max + self.scale, self.scale)
                self.size = len(self.range)

    @staticmethod
    def scalar_to_vector(x, **kwargs):
        env = kwargs['env']
        x_min = kwargs['range'][0]
        x_max = kwargs['range'][1]
        epsilon = 0.001

        if env in ['DMC', 'Gym']:
            x_min = transform_one(x_min)
            x_max = transform_one(x_max)
            bins = kwargs['bins']
            scale = (x_max - x_min) / (bins - 1)
            x_range = np.arange(x_min, x_max + scale, scale)
            sign = torch.ones(x.shape).float().to(x.device)
            sign[x < 0] = -1.0
            x = sign * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x
            x = x / scale

            x.clamp_(x_min / scale, x_max / scale - 1e-5)
            x = x - x_min / scale
            x_low_idx = x.floor()
            x_high_idx = x.ceil()
            p_high = x - x_low_idx
            p_low = 1 - p_high

            target = torch.zeros(tuple(x.shape) + (bins,), dtype=p_high.dtype).to(x.device)
            target.scatter_(len(x.shape), x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
            target.scatter_(len(x.shape), x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        else:
            scale = kwargs['scale']
            x_range = np.arange(x_min, x_max + scale, scale)
            x_size = len(x_range)

            sign = torch.ones(x.shape).float().to(x.device)
            sign[x < 0] = -1.0
            x = sign * (torch.sqrt(torch.abs(x / scale) + 1) - 1 + epsilon * x / scale)

            x.clamp_(x_min, x_max)
            x_low = x.floor()
            x_high = x.ceil()
            p_high = x - x_low
            p_low = 1 - p_high

            target = torch.zeros(x.shape[0], x.shape[1], x_size).to(x.device)
            x_high_idx, x_low_idx = x_high - x_min / scale, x_low - x_min / scale
            target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
            target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target

    @staticmethod
    def vector_to_scalar(logits, **kwargs):
        x_min = kwargs['range'][0]
        x_max = kwargs['range'][1]
        env = kwargs['env']
        epsilon = 0.001

        if env in ['DMC', 'Gym']:
            x_min = transform_one(x_min)
            x_max = transform_one(x_max)
            bins = kwargs['bins']
            scale = (x_max - x_min) / (bins - 1)
            x_range = np.arange(x_min, x_max + scale, scale)

            value_probs = torch.softmax(logits, dim=-1)
            value_support = torch.ones(value_probs.shape)
            value_support[:, :] = torch.from_numpy(np.array([x for x in x_range]))
            value_support = value_support.to(device=value_probs.device)
            value = (value_support * value_probs).sum(-1, keepdim=True) / scale

            sign = torch.ones(value.shape).float().to(value.device)
            sign[value < 0] = -1.0
            output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) * scale + 1 + epsilon)) - 1) / (
                    2 * epsilon)) ** 2 - 1)
            output = sign * output
        else:
            scale = kwargs['scale']
            x_range = np.arange(x_min, x_max + scale, scale)
            value_probs = torch.softmax(logits, dim=-1)
            value_support = torch.ones(value_probs.shape)
            value_support[:, :] = torch.from_numpy(np.array([x for x in x_range]))
            value_support = value_support.to(device=value_probs.device)
            value = (value_support * value_probs).sum(-1, keepdim=True) / scale

            sign = torch.ones(value.shape).float().to(value.device)
            sign[value < 0] = -1.0
            output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
            output = sign * output * scale

            nan_part = torch.isnan(output)
            output[nan_part] = 0.
            output[torch.abs(output) < epsilon] = 0.
        return output


def formalize_obs_lst(obs_lst, already_prepare=False):
    obs_lst = np.asarray(obs_lst)
    obs_lst = torch.from_numpy(obs_lst).cuda().float()
    shape = obs_lst.shape
    obs_lst = obs_lst.reshape((shape[0], -1))
    return obs_lst


def prepare_obs_lst(obs_lst):
    """Prepare the observations to satisfy the input format of torch
    [B, S, H] -> [B, S x H]
    batch, stack num, hidden size
    """
    obs_lst = np.asarray(obs_lst)
    shape = obs_lst.shape
    obs_lst = obs_lst.reshape((shape[0], -1))
    return obs_lst


def normalize_state(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min) / (max - min)
    return flat_tensor.view(*tensor.shape)


def softmax(logits):
    logits = np.asarray(logits)
    logits -= logits.max()
    logits = np.exp(logits)
    logits = logits / logits.sum()
    return logits


def pad_and_mask(trajectories, pad_value=0, is_action=False):
    """Pads the trajectories to the same length and creates an attention mask."""
    max_len = max([len(t) for t in trajectories])
    masks = torch.ones(len(trajectories), max_len).cuda().bool()
    
    padded_trajectories = []
    for i, traj in enumerate(trajectories):
        if is_action:
            padded_traj = torch.nn.functional.pad(traj, (0, max_len - len(traj)), value=pad_value)
        else:
            padded_traj = torch.nn.functional.pad(traj, (0, 0, 0, 0, 0, 0, 0, max_len - len(traj)), value=pad_value)
        masks[i, :len(traj)] = False
        padded_trajectories.append(padded_traj)

    try:
        padded_trajectories = torch.stack(padded_trajectories)
    except:
        import ipdb
        ipdb.set_trace()
        print('false')
    return padded_trajectories, masks


def init_logger(base_path):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['Train', 'Eval']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


def get_ddp_model_weights(ddp_model):
    """Get weights of a DDP model"""
    return {'.'.join(k.split('.')[2:]): v.cpu() for k, v in ddp_model.state_dict().items()}


def allocate_gpu(rank, gpu_lst, worker_name):
    time.sleep(3)
    available_memory_list = get_gpu_memory()
    for i in range(len(available_memory_list)):
        if i not in gpu_lst:
            available_memory_list[i] = -1
    available_memory_list[0] -= 4000  # avoid using gpu 0, which is left for training
    available_memory_list[1] -= 6000  # avoid using gpu 1, which is left for training
    max_index = available_memory_list.index(max(available_memory_list))
    if available_memory_list[max_index] < 2000:
        print(f"[{worker_name} worker GPU]******************* Warning: Low video ram (max remaining "
              f"{available_memory_list[max_index]}) *******************")
    torch.cuda.set_device(max_index)
    print(f"[{worker_name} worker GPU] {worker_name} worker GPU {rank} at process {os.getpid()}"
          f" will use GPU {max_index}. Remaining memory before allocation {available_memory_list}")


def get_gpu_memory():
    """Returns available gpu memory for each available gpu"""
    def _output_to_list(x):
        return x.decode('ascii').split('\n')[:-1]
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(command.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def profile(func):
    from line_profiler import LineProfiler
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        result = lp_wrapper(*args, **kwargs)
        lp.print_stats()
        return result
    return wrapper
