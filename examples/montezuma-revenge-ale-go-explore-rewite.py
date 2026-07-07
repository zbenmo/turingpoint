from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import functools
import math
import pickle
# from pathlib import Path
import itertools
import random
from typing import Any, Dict, TypeAlias
import numpy as np
import hashlib
import ale_py
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as dist
import matplotlib.pyplot as plt
import gc
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

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


# copied from clearRL
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNLayers(nn.Module):
    """CNN layers after each there is a non-linearity (ReLU), and also a flattening at the end."""
    def __init__(self, in_channels, activ=nn.ReLU, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cnn_layers = []
        in_channels = in_channels  # for 4 frames (history), 1 for a single frame
        out_size = np.array((84, 84))
        for kernel_size, stride, out_channels in zip([8, 4, 3], [4, 2, 1], [32, 64, 64]):
            cnn_layers.append(
                layer_init(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)))
            cnn_layers.append(activ())
            in_channels = out_channels
            out_size = (out_size - kernel_size) // stride + 1
        self.num_ele = (out_channels * out_size.prod()).item()
        assert self.num_ele == 3136, f'{out_channels=}, {out_size=}'
        self.net = nn.Sequential(
            *cnn_layers,
            nn.Flatten(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> value (a regression)"""

        return self.net(obs)

    @property
    def num_elements(self) -> int:
        return self.num_ele


class StateToActionLogits(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_layers = CNNLayers(in_channels=4)
        self.net = nn.Sequential(
            self.cnn_layers,
            layer_init(nn.Linear(3136, 512), std=0.1),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            layer_init(nn.Linear(512, out_features.item()), std=0.1),
        )
        self.out_features = out_features  # keep for the range of actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> actions (logits)"""

        return self.net(obs)


def _clone_state(env):
    return env.unwrapped.clone_state()


def restore_and_observe(env, state):
    env.unwrapped.restore_state(state)
    # Rebuild wrapper buffers
    obs, _, _, _, _ = env.step(0)  # NOOP
    return obs


class ResumeFromStateEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._restore_state: State | None = None
        self._pending_restore = False

    def set_restore_state(self, state: State):
        self._restore_state = state
        self._pending_restore = True

    def reset(self, *, seed=None, options=None):
        if self._pending_restore and self._restore_state is not None:
            obs = restore_and_observe(self.env, self._restore_state)
            self._pending_restore = False
            return obs, {}
        return super().reset(seed=seed, options=options)


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
        ax.set_title(f"pos {idx}", fontsize=8)
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


def action_to_symbol(action: Action) -> str:
    mapping = {
        0: "NOPE",
        1: "FIRE",
        2: "↑",
        3: "→",
        4: "←",
        5: "↓",
        6: "↗",
        7: "↖",
        8: "↘",
        9: "↙",
        10: "↑FIRE",
        11: "→FIRE",
        12: "←FIRE",
        13: "↓FIRE",
        14: "↗FIRE",
        15: "↖FIRE",
        16: "↘FIRE",
        17: "↙FIRE",
    }
    return mapping.get(action, str(action))


def plot_trajectory_segment(
    env,
    states: list[State],
    actions: list[Action],
    rewards: list[float],
    start: int,
    end: int,
    filename: str | None = None,
):
    num_frames = end - start + 1
    if num_frames <= 0:
        return

    grid_size = int(math.ceil(math.sqrt(num_frames)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
    if grid_size == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    for plot_idx, idx in enumerate(range(start, end + 1)):
        ax = axes_flat[plot_idx]
        _ = restore_and_observe(env, states[idx])
        frame = env.unwrapped.ale.getScreenRGB()

        ax.imshow(frame)
        ax.set_title(f"{idx}: {action_to_symbol(actions[idx])}, r={rewards[idx]}", fontsize=8)
        ax.axis("off")

    for extra_ax in axes_flat[num_frames:]:
        extra_ax.axis("off")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        print(f"Saved trajectory segment visualization to {filename}")
    try:
        plt.show()
    except Exception as e:
        print(f"Could not run plt.show(): {e}")


def save_trajectory_video(env, trajectory: list[Any], output_path: str = "best_path.mp4"):
    if not trajectory:
        return

    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    frames = []
    for state in trajectory:
        _ = restore_and_observe(env, state)
        frames.append(env.unwrapped.ale.getScreenRGB())

    clip = ImageSequenceClip(frames, fps=20)
    clip.write_videofile(output_path, codec="libx264", audio=False)
    print(f"Saved trajectory video to {output_path}")


def train(env, exploration_steps: int):

    key_not_in_archive_count: int = 0
    bigger_reward_count: int = 0
    shorter_trajectory_count: int = 0

    @dataclass(frozen=True)
    class ArchiveItem:
        state: State
        parent: ArchiveItem | None = None
        action: Action | None = None # from parent here
        reward: float = 0 # by reaching here
        terminated: bool = False # AKA done
        total_reward: float = 0
        trajectory_len: int = 0

    archive: dict[str, ArchiveItem] = dict()
    counts = Counter()

    def init_exploration(parcel: Dict):
        state = _clone_state(env)
        obs = parcel['obs']
        key = obs_to_key(obs)
        archive[key] = ArchiveItem(state=state)
        parcel['parent_entry'] = archive[key]
        counts[key] += 1

    def consider_new_state(parcel: Dict, env):
        nonlocal key_not_in_archive_count
        nonlocal bigger_reward_count
        nonlocal shorter_trajectory_count

        obs = parcel['obs']
        action = parcel['action']
        key = obs_to_key(obs)
        counts[key] += 1
        reward = parcel['reward']
        if reward > 0:
            print(f'{reward=}')
        terminated = parcel['terminated']

        parent_entry = parcel.pop('parent_entry', None)
        assert parent_entry is not None
        total_reward = parent_entry.total_reward + reward
        trajectory_len = parent_entry.trajectory_len + 1

        newItem = ArchiveItem(
            state=_clone_state(env),
            parent=parent_entry,
            action=action,
            reward=reward,
            terminated=terminated,
            total_reward=total_reward,
            trajectory_len=trajectory_len
        )

        key_not_in_archive = key not in archive
        bigger_reward = (key in archive) and total_reward > archive[key].total_reward
        # shorter_trajectory = (key in archive) and ( # and (parcel.get('step', 0) >  100_000) 
        #     total_reward == archive[key].total_reward
        #     and trajectory_len < archive[key].trajectory_len
        # )
        shorter_trajectory = False

        if key_not_in_archive:
            key_not_in_archive_count += 1
        elif bigger_reward:
            bigger_reward_count += 1
        elif shorter_trajectory:
            shorter_trajectory_count += 1
            # print(f"shorter_trajectory: {total_reward=}, {trajectory_len=}, { archive[key].trajectory_len=}")

        if (
            key_not_in_archive
            or bigger_reward
            or shorter_trajectory
        ):
            archive[key] = newItem

        parcel['parent_entry'] = newItem


    def score(key: str, parcel: Dict) -> float: # my interpretation
        exploitation = archive[key].total_reward / 200
        step: float = parcel.get('step', 0) + 1
        exploration = math.sqrt(2 * math.log2(step) / counts[key])
        return exploitation + exploration


    def select_an_entry_to_explore_further(parcel: Dict):
        selected_key = max(archive, key=functools.partial(score, parcel=parcel))
        selected_entry = archive[selected_key]
        parcel['obs'] = restore_and_observe(env, selected_entry.state)
        parcel['parent_entry'] = selected_entry
        # TODO: restore reward, terminated, truncated
        counts[selected_key] += 1


    def check_if_time_to_reset(parcel: Dict):
        if not (
            parcel.get('terminated', False)
            or parcel.get('truncated', False)
            or parcel.get('step', 0) % 200 == 0
        ):
            return

        select_an_entry_to_explore_further(parcel)


    def explore():

        def get_participants_exploration():
            with tp_utils.StepsTracker(exploration_steps, desc="exploration steps") as steps_tracker:
                yield functools.partial(tp_gym_utils.call_reset, env=env)
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

    def robustify(trajectory: list[tuple[Observation, Action]]) -> StateToActionLogits:
        obs, action = zip(*trajectory)

        obs = obs[:-1]
        action = action[1:]

        state_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        agent = StateToActionLogits(state_space, action_space)

        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
        actions_tensor = torch.tensor(action, dtype=torch.long)

        ds = TensorDataset(obs_tensor, actions_tensor)
        dl = DataLoader(ds, batch_size=256, shuffle=True)

        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

        for epoch in range(1, 400 + 1):
            losses = []
            for o, act in dl:

                logits = agent(o)

                optimizer.zero_grad()

                loss = F.cross_entropy(logits, act)

                losses.append(loss.item())
                loss.backward()

                optimizer.step()

            if epoch % 10 == 0:
                print(f'epoch={epoch}, mean loss={np.mean(losses)}')

        return agent


    def robustify_PPO(trajectory: list[tuple[Observation, Action]]) -> PPO:
        obs, action = zip(*trajectory)

        obs = obs[:-1]
        action = action[1:]

        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
        actions_tensor = torch.tensor(action, dtype=torch.long)

        ds = TensorDataset(obs_tensor, actions_tensor)
        dl = DataLoader(ds, batch_size=256, shuffle=True)

        model = PPO("CnnPolicy", env, verbose=1)
        policy = model.policy

        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        # loss_fn = nn.CrossEntropyLoss()

        for epoch in range(200):
            losses = []
            for batch_obs, batch_actions in dl:

                # 1. Extract features from the CNN
                latent_pi = policy.extract_features(batch_obs)

                # 2. Build the action distribution
                dist = policy._get_action_dist_from_latent(latent_pi)

                # 3. Get raw logits
                logits = dist.distribution.logits

                # 4. Cross entropy loss
                loss = F.cross_entropy(logits, batch_actions)

                losses.append(loss.item())

                # logits = policy(batch_obs)[0]  # policy returns (logits, value)
                # loss = loss_fn(logits, batch_actions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"{epoch} -> loss = {np.mean(losses)}")

        return model

    class RewardTracker(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.current_reward = 0

        def _on_step(self) -> bool:
            # SB3 stores episode end in info["episode"]
            info = self.locals["infos"][0]
            reward = self.locals["rewards"][0]

            self.current_reward += reward

            if "episode" in info:
                # Episode finished
                self.episode_rewards.append(self.current_reward)
                self.current_reward = 0

            return True

    def robustify_by_retreat(trajectory: list[tuple[Observation, Action]]) -> PPO:
        state, action, reward, terminated = zip(*trajectory)

        obs = [restore_and_observe(env, s) for s in state]

        # model = robustify_PPO(list(zip(obs, action)))

        obs, next_obs = obs[:-1], obs[1:]
        action = action[1:]
        reward = reward[1:]
        terminated = terminated[1:]
        state = state[:-1]

        assert len(next_obs) == len(obs)
        assert len(action) == len(obs)
        assert len(reward) == len(obs)
        assert len(terminated) == len(obs)
        assert len(state) == len(obs)

        with_reward = [i for i in range(len(reward)) if reward[i] > 0]

        resume_env = ResumeFromStateEnv(env)
        model = PPO("CnnPolicy", resume_env, verbose=1)

        # print(','.join(map(str, with_reward)))

        # exit(0)

        go_back = 2

        for ind in reversed(with_reward):
            start = max(0, ind - go_back)

            # show the states from here to the ind (included), with the reward(s) and the actions
            plot_trajectory_segment(env, state, action, reward, start, ind, filename=f"retreat_segment_{start}_{ind}.png")

            print(f'{start=}, {ind=}')
            while True:
                print('.')
                resume_env.set_restore_state(state[start])
                tracker = RewardTracker()
                model.learn(100, callback=tracker)
                if sum(tracker.episode_rewards) > 0:
                    break
        # print(len(obs))

        # obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
        # actions_tensor = torch.tensor(action, dtype=torch.long)

        # ds = TensorDataset(obs_tensor, actions_tensor)
        # dl = DataLoader(ds, batch_size=256, shuffle=True)

        # policy = model.policy

        # optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        # loss_fn = nn.CrossEntropyLoss()

        # for epoch in range(200):
        #     for batch_obs, batch_actions in dl:

        #         # 1. Extract features from the CNN
        #         latent_pi = policy.extract_features(batch_obs)

        #         # 2. Build the action distribution
        #         dist = policy._get_action_dist_from_latent(latent_pi)

        #         # 3. Get raw logits
        #         logits = dist.distribution.logits

        #         # 4. Cross entropy loss
        #         loss = F.cross_entropy(logits, batch_actions)

        #         # logits = policy(batch_obs)[0]  # policy returns (logits, value)
        #         # loss = loss_fn(logits, batch_actions)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        return model


    if False:

        explore()

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

        if True:
            trajectory_states = [x.state for x in trajectory]
            plot_best_path(env, trajectory_states)
            save_trajectory_video(env, trajectory_states)

        print(f'{key_not_in_archive_count=}')
        print(f'{bigger_reward_count=}')
        print(f'{shorter_trajectory_count=}')

        archive.clear()
        counts.clear()
        gc.collect()

        # save trajectory to trajectory.save
        trajectory_for_save = [
            (
                archiveItem.state, # restore_and_observe(env, archiveItem.state),
                archiveItem.action,
                archiveItem.reward,
                archiveItem.terminated,
            )
            for archiveItem in trajectory
        ]
        pickle.dump(trajectory_for_save, open("trajectory.save", "wb"))

    if True:

        # load trajectory from trajectory.save
        trajectory = pickle.load(open("trajectory.save", "rb"))

        # agent = robustify(trajectory)
        # agent = robustify_PPO(trajectory)
        agent = robustify_by_retreat(trajectory)

        obs, _ = env.reset(seed=0)
        trajectory_states = [_clone_state(env)]

        for _ in range(1_000):
            action, _ = agent.predict(obs, deterministic=True)
            next_obs, _, terminated, truncated, _ = env.step(action)
            trajectory_states.append(_clone_state(env))
            obs = next_obs
            if terminated or truncated:
                break

        plot_best_path(env, trajectory_states)
        save_trajectory_video(env, trajectory_states)


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

    exploration_steps = 500_000  # Adjust as needed

    train(env, exploration_steps)


if __name__ == "__main__":
    main()