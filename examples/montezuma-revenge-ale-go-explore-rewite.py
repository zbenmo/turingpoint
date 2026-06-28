from collections import Counter, defaultdict
from dataclasses import dataclass
import functools
import math
from pathlib import Path
import itertools
import random
from typing import Any, Dict
import numpy as np
import hashlib
import ale_py
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import moviepy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as dist
import matplotlib.pyplot as plt
import gc

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint as tp

# add device selection and CUDA tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

gym_environment = "ALE/MontezumaRevenge-v5"


def _clone_state(env):
    return env.unwrapped.clone_state()


def clone_state(parcel: Dict, env, save_state_as: str = 'state'):
    parcel[save_state_as] = _clone_state(env)


def restore_and_observe(env, state):
    env.unwrapped.restore_state(state)
    # Rebuild wrapper buffers
    obs, _, _, _, _ = env.step(0)  # NOOP
    return obs


def get_random_action(parcel: Dict, env, repeat_prob: float | None = None):
    if (
        parcel.get('action') is not None
        and repeat_prob is not None
        and random.random() < repeat_prob
    ):
        return
    parcel['action'] = env.action_space.sample()


def advance(parcel: Dict):
    parcel['obs'] = parcel.pop('new_obs')


def downsample_obs(obs: np.ndarray) -> np.ndarray:
    """Hash the observation for archiving. Use the last frame for hashing, downsampled to 11x8 as per Go-Explore paper."""
    last_frame = obs[-1]  # shape (84, 84), uint8, 0-255
    # Convert to tensor for interpolation
    frame_tensor = torch.tensor(last_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,84,84)
    # Downsample to 11x8 using area interpolation (averaging)
    downsampled = F.interpolate(frame_tensor, size=(11, 8), mode='area')  # (1,1,11,8)
    downsampled = downsampled.squeeze().numpy()  # (11,8)
    # Rescale to 0-8 integers
    return np.clip(downsampled / 255.0 * 8, 0, 8).astype(np.uint8)


def obs_to_key(obs: np.ndarray) -> str:
    downsampled = downsample_obs(obs)
    return hashlib.sha256(downsampled.tobytes()).hexdigest()


def train(env, exploration_steps: int):

    @dataclass
    class ArchiveItem:
        state: Any
        path: list[str]
        total_reward: float

    archive: dict[str, ArchiveItem] = dict()
    counts = Counter()

    def init_exploration(parcel: Dict):
        state = parcel.pop('state')
        obs = parcel['obs']
        key = obs_to_key(obs)
        archive[key] = ArchiveItem(state=state, path=[], total_reward=0)
        parcel['parent_key'] = key
        counts[key] += 1
        # print(counts)

    def consider_new_state(parcel: Dict, env):
        obs = parcel['obs']
        key = obs_to_key(obs)
        counts[key] += 1
        # print(counts)
        reward = parcel['reward']
        if reward > 0:
            print(f'{reward=}')
        parent_key = parcel.pop('parent_key', None)
        if parent_key:
            parent_entry = archive[parent_key]
            total_reward = parent_entry.total_reward + reward
            new_path = parent_entry.path[:]
            new_path.append(parent_key)
        else:
            total_reward = reward
            new_path = []
        if (
            key not in archive
            or total_reward > archive[key].total_reward
            or total_reward == archive[key].total_reward and len(new_path) < len(archive[key].path)
        ):
            state = _clone_state(env)
            archive[key] = ArchiveItem(state=state, path=new_path, total_reward=total_reward)
        # TODO: decide if to replace, maintain total reward
        parcel['parent_key'] = key


    def check_if_time_to_reset(parcel: Dict):
        if not (
            parcel.get('terminated', False)
            or parcel.get('truncated', False)
            or parcel.get('step', 0) % 200 == 0
        ):
            # print(f'step={parcel.get("step")}')
            return

        s = sorted(archive, key=lambda key: counts[key]) # TEMP TODO:
        selected_key = s[0]
        # print(f"counter={counts[selected_key]}")
        # print(f"archive[selected_key]={archive[selected_key]}")
        parcel['obs'] = restore_and_observe(env, archive[selected_key].state)
        parcel.pop('parent_key', None)
        if len(archive[selected_key].path) > 0:
            parcel['parent_key'] = archive[selected_key].path[-1]
        # TODO: restore reward, terminated, truncated
        counts[selected_key] += 1

    def explore():

        # collector = tp_utils.Collector(['state', 'obs', 'action', 'reward', 'new_obs'])

        def get_participants_exploration():
            with tp_utils.StepsTracker(exploration_steps, desc="exploration steps") as steps_tracker:
                yield functools.partial(tp_gym_utils.call_reset, env=env)
                yield functools.partial(clone_state, env=env) # TODO: move it into init_exploration
                yield init_exploration
                yield from itertools.cycle([
                    functools.partial(get_random_action, env=env, repeat_prob=0.4),
                    functools.partial(tp_gym_utils.call_step, env=env, save_obs_as='new_obs'),
                    functools.partial(consider_new_state, env=env),
                    # functools.partial(clone_state, env=env, save_state_as='new_state'),
                    # collector,
                    advance,
                    steps_tracker,
                    check_if_time_to_reset
                ])

        exploration_assembly = tp.Assembly(get_participants_exploration)
        _ = exploration_assembly.launch()

        print(f'archive len={len(archive)}')
        print(f'counts len={len(counts)}')


    def robustify():
        pass

    explore()
    robustify()


def make_env(seed, **kwargs) -> gym.Env:
    env = gym.make(gym_environment, frameskip=4, **kwargs)  # (210, 160, 3)
    env.action_space.seed(seed=seed)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=1,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    ) # (84, 84)
    env = FrameStackObservation(env, stack_size=4) # (4, 84, 84)
    env.reset(seed=seed)
    return env


def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    gym.register_envs(ale_py)

    env = make_env(seed=0)

    exploration_steps = 200_000  # Adjust as needed

    train(env, exploration_steps)


if __name__ == "__main__":
    main()