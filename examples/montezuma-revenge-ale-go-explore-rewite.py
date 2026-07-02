from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import functools
import math
from pathlib import Path
import itertools
import random
from typing import Any, Dict, TypeAlias
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

Observation: TypeAlias = np.ndarray
Action: TypeAlias = int
State: TypeAlias = Any

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


def downsample_obs(obs: Observation) -> np.ndarray:
    """Hash the observation for archiving. Use the last frame for hashing, downsampled to 11x8 as per Go-Explore paper."""
    last_frame = obs[-1]  # shape (84, 84), uint8, 0-255
    # Convert to tensor for interpolation
    frame_tensor = torch.tensor(last_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,84,84)
    # Downsample to 11x8 using area interpolation (averaging)
    downsampled = F.interpolate(frame_tensor, size=(11, 8), mode='area')  # (1,1,11,8)
    downsampled = downsampled.squeeze().numpy()  # (11,8)
    # Rescale to 0-8 integers
    return np.clip(downsampled / 255.0 * 8, 0, 8).astype(np.uint8)


def obs_to_key(obs: Observation) -> str:
    downsampled = downsample_obs(obs)
    return hashlib.sha256(downsampled.tobytes()).hexdigest()


def plot_best_path(env, best_path: list[Any]):
    pass
    # best_key = max(seen_obs.keys(), key=lambda k: (seen_obs[k].cumulative_reward, seen_obs[k].step_count))
    # best_entry = seen_obs[best_key]
    # print(f"Best path cumulative reward: {best_entry.cumulative_reward}, step count: {best_entry.step_count}")
    
    # # Traceback
    # path = []
    # current_key = best_key
    # while current_key is not None:
    #     entry = seen_obs[current_key]
    #     if entry.parent_key in set(x[0] for x in path):
    #         break
    #     path.append((current_key, entry))
    #     current_key = entry.parent_key
    # path.reverse()
    
    print(f"Path length (actual saved cells): {len(best_path)}")
    
    # Select up to 16 states to plot
    num_to_plot = min(16, len(best_path))
    if num_to_plot == 0:
        return
        
    # Pick indices evenly spaced
    indices = np.linspace(0, len(best_path) - 1, num_to_plot, dtype=int)
    
    # Determine grid size
    grid_size = int(math.ceil(math.sqrt(num_to_plot)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    
    # Flatten axes for easy iteration
    if grid_size == 1:
        axes = np.array([axes])
    axes_flat = axes.flat
    
    for i, idx in enumerate(indices):
        ax = axes_flat[i]
        state = best_path[idx]
        
        # Restore state and render screen
        _ = restore_and_observe(env,state)
        obs = env.unwrapped.ale.getScreenRGB()
        
        ax.imshow(obs)
        # ax.set_title(f"Step: {entry.step_count}\nRew: {entry.cumulative_reward}", fontsize=8)
        ax.axis('off')
        
    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
        
    plt.tight_layout()
    plt.savefig("best_path.png")
    print("Saved best path visualization to best_path.png")
    try:
        plt.show()
    except Exception as e:
        print(f"Could not run plt.show(): {e}")


def train(env, exploration_steps: int):

    @dataclass(frozen=True)
    class ArchiveItem:
        state: State
        parent: ArchiveItem | None = None
        action: Action | None = None # from parent here
        total_reward: float = 0

    archive: dict[str, ArchiveItem] = dict()
    counts = Counter()

    def init_exploration(parcel: Dict):
        state = parcel.pop('state')
        obs = parcel['obs']
        key = obs_to_key(obs)
        archive[key] = ArchiveItem(state=state)
        parcel['parent_entry'] = archive[key]
        counts[key] += 1

    def consider_new_state(parcel: Dict, env):
        obs = parcel['obs']
        action = parcel['action']
        key = obs_to_key(obs)
        counts[key] += 1
        reward = parcel['reward']
        if reward > 0:
            print(f'{reward=}')

        parent_entry = parcel.pop('parent_entry', None)
        if parent_entry:
            total_reward = parent_entry.total_reward + reward
        else:
            total_reward = reward

        if (
            key not in archive
            or total_reward > archive[key].total_reward
            # or (
            #     total_reward == archive[key].total_reward
            #     and len(trajectory) < len(archive[key].trajectory)
            # )
        ):
            state = _clone_state(env)
            archive[key] = ArchiveItem(
                state=state,
                parent=parent_entry,
                action=action,
                total_reward=total_reward
            )

        parcel['parent_entry'] = archive[key]


    def check_if_time_to_reset(parcel: Dict):
        if not (
            parcel.get('terminated', False)
            or parcel.get('truncated', False)
            or parcel.get('step', 0) % 200 == 0
        ):
            return

        s = sorted(archive, key=lambda key: counts[key]) # TEMP TODO:
        selected_key = s[0]
        parcel['obs'] = restore_and_observe(env, archive[selected_key].state)
        parcel.pop('parent_entry', None)
        if archive[selected_key].parent is not None:
            parcel['parent_entry'] = archive[selected_key].parent
        # TODO: restore reward, terminated, truncated
        counts[selected_key] += 1


    def explore():

        def get_participants_exploration():
            with tp_utils.StepsTracker(exploration_steps, desc="exploration steps") as steps_tracker:
                yield functools.partial(tp_gym_utils.call_reset, env=env)
                yield functools.partial(clone_state, env=env) # TODO: move it into init_exploration
                yield init_exploration
                yield from itertools.cycle([
                    functools.partial(get_random_action, env=env, repeat_prob=0.4),
                    functools.partial(tp_gym_utils.call_step, env=env, save_obs_as='new_obs'),
                    functools.partial(consider_new_state, env=env),
                    advance,
                    steps_tracker,
                    check_if_time_to_reset
                ])

        exploration_assembly = tp.Assembly(get_participants_exploration)
        _ = exploration_assembly.launch()

        print(f'archive len={len(archive)}')
        print(f'counts len={len(counts)}')

        s = sorted(archive, key=lambda key: archive[key].total_reward)
        best = archive[s[-1]]

        trajectory = [best]
        p = best.parent
        while p is not None:
            trajectory.append(p)
            p = p.parent
        trajectory.reverse()

        plot_best_path(env, [x.state for x in trajectory])


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