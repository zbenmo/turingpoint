from collections import defaultdict
import functools
import math
from pathlib import Path
import itertools
import random
from typing import Any, Dict, List
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

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint as tp

# add device selection and CUDA tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

gym_environment = "ALE/MontezumaRevenge-v5"

# # Hyperparameters for cell scoring (Section A.5)
# EPSILON_1 = 0.001  # Prevents division by 0, determines weight for 0 values
# EPSILON_2 = 0.00001  # Ensures no cell has 0 probability
# CELL_SCORE_WEIGHTS = {
#     'num_times_chosen': 1.0,           # wa for times chosen
#     'num_times_visited': 0.5,         # wa for times visited
#     'num_times_chosen_since_improvement': 2.0  # wa for times chosen since improvement
# }
# CELL_SCORE_POWERS = {
#     'num_times_chosen': 1.0,
#     'num_times_visited': 1.0,
#     'num_times_chosen_since_improvement': 1.0
# }

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

def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """Scale observations."""
    return obs.astype(np.float32) / 255.0

def make_env(seed, **kwargs) -> gym.Env:
    env = gym.make(gym_environment, frameskip=4, **kwargs)  # (210, 160, 3)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=1,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)
    return env

videos_path = Path("videos")
videos_path.mkdir(exist_ok=True)

def record_video(env, agent, video_path: Path):
    """Records a video of the agent playing."""
    def get_participants():
        yield functools.partial(tp_gym_utils.call_reset, env=env)
        yield from itertools.cycle([
            functools.partial(get_action_policy, agent=agent),
            functools.partial(tp_gym_utils.call_step, env=env),
            tp_gym_utils.check_done
        ])

    assembly = tp.Assembly(get_participants)
    assembly.launch()

    frames = env.render()
    clip = moviepy.ImageSequenceClip([np.uint8(frame) for frame in frames], fps=env.metadata["render_fps"])
    clip.write_videofile(str(video_path), codec="libx264", logger=None)

# def hash_observation(obs: np.ndarray) -> str:
#     """Hash the observation for archiving. Use the last frame for hashing, downsampled to 11x8 as per Go-Explore paper."""
#     last_frame = obs[-1]  # shape (84, 84), uint8, 0-255
#     # Convert to tensor for interpolation
#     frame_tensor = torch.tensor(last_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,84,84)
#     # Downsample to 11x8 using area interpolation (averaging)
#     downsampled = F.interpolate(frame_tensor, size=(11, 8), mode='area')  # (1,1,11,8)
#     downsampled = downsampled.squeeze().numpy()  # (11,8)
#     # Rescale to 0-8 integers
#     downsampled = np.clip(downsampled / 255.0 * 8, 0, 8).astype(np.uint8)
#     return hashlib.md5(downsampled.tobytes()).hexdigest()

# def call_reset_done_single(parcel, *, env: gym.Env, **kwargs):
#     if parcel.get('terminated', False) or parcel.get('truncated', False):
#         ob, inf = env.reset(**kwargs)
#         parcel['obs'] = ob
#         parcel['info'] = inf

def get_random_action(parcel: Dict, env):
    """Picks a random action."""

    if parcel.get('action') and random.random() > 0.3: # TODO: decide < 0.3 or > 0.3, use repeat_prob = min(0.95, 0.2 + 0.000001 * total_steps)
        return
    parcel['action'] = env.action_space.sample()

def get_action_policy(parcel: Dict, *, agent: StateToActionLogits):
    obs = parcel['obs']
    obs = preprocess_observation(obs)
    if len(obs.shape) == 4:
        obs_tensor = torch.tensor(obs)
        obs_tensor = obs_tensor.reshape(-1, *obs_tensor.shape[-3:])
    else:
        obs_tensor = torch.tensor(obs).unsqueeze(0)
    with torch.no_grad():
        logits = agent(obs_tensor)
        action_dist = dist.Categorical(logits=logits)
        action = action_dist.sample()
        parcel['action'] = action.cpu().numpy().item() if action.numel() == 1 else action.cpu().numpy()

# def archive_state(parcel: Dict, *, archive: Dict, env: gym.Env):
#     """Archive the current observation if novel, tracking visit counts."""
#     obs = parcel['obs']
#     reward = parcel.get('reward', 0)
    
#     if len(obs.shape) == 4:  # vectorized
#         for o in obs:
#             h = hash_observation(o)
#             if h not in archive:
#                 # Clone the full environment state for proper resetting
#                 state = env.unwrapped.clone_state()
#                 archive[h] = {
#                     'obs': o.copy(),
#                     'state': state,
#                     'num_times_chosen': 0,
#                     'num_times_visited': 0,
#                     'num_times_chosen_since_improvement': 0,
#                     'best_reward_seen': 0,
#                 }
#             archive[h]['num_times_visited'] += 1
#             archive[h]['best_reward_seen'] = max(archive[h].get('best_reward_seen', 0), reward if isinstance(reward, (int, float)) else 0)
#     else:
#         h = hash_observation(obs)
#         if h not in archive:
#             # Clone the full environment state for proper resetting
#             state = env.unwrapped.clone_state()
#             archive[h] = {
#                 'obs': obs.copy(),
#                 'state': state,
#                 'num_times_chosen': 0,
#                 'num_times_visited': 0,
#                 'num_times_chosen_since_improvement': 0,
#                 'best_reward_seen': 0,
#             }
#         archive[h]['num_times_visited'] += 1
#         archive[h]['best_reward_seen'] = max(archive[h].get('best_reward_seen', 0), reward if isinstance(reward, (int, float)) else 0)

# def collect_trajectory_for_bc(parcel: Dict, *, trajectories: List, archive: Dict, current_trajectory: List):
#     """Collect trajectory data for BC."""
#     obs = parcel['obs']
#     action = parcel['action']
    
#     if len(obs.shape) == 4:
#         obs = obs[0]
#         action = action[0]
    
#     current_trajectory.append({'obs': obs.copy(), 'action': action})
#     h = hash_observation(obs)
#     if h in archive and len(trajectories) < 100:
#         # If we reached an archived state, save the trajectory
#         trajectories.append({'steps': current_trajectory.copy()})
#         current_trajectory.clear()

# def train_bc_policy(trajectories: List, policy: StateToActionLogits, optimizer, num_epochs=10):
#     """Train policy using Behavioral Cloning on collected trajectories."""
#     if not trajectories:
#         return
    
#     # Use all trajectories for training
#     top_trajectories = trajectories
    
#     if top_trajectories:
#         print(f"Training BC on {len(top_trajectories)} trajectories")
    
#     # Flatten trajectories into (obs, action) pairs
#     obs_list = []
#     action_list = []
#     for traj_data in top_trajectories:
#         traj = traj_data.get('steps', traj_data)
#         for step in traj:
#             obs_list.append(preprocess_observation(step['obs']))
#             action_list.append(step['action'])

#     obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)
#     action_tensor = torch.tensor(np.array(action_list), dtype=torch.long).to(device)

#     dataset = TensorDataset(obs_tensor, action_tensor)
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#     policy.train()
#     for epoch in range(num_epochs):
#         for obs_batch, action_batch in dataloader:
#             optimizer.zero_grad()
#             logits = policy(obs_batch)
#             loss = F.cross_entropy(logits, action_batch)
#             loss.backward()
#             optimizer.step()

# def calculate_cell_score(cell_data: Dict) -> float:
#     """Calculate score for a cell based on count subscores (Section A.5)."""
#     score = 0.0
#     attributes = {
#         'num_times_chosen': cell_data.get('num_times_chosen', 0),
#         'num_times_visited': cell_data.get('num_times_visited', 0),
#         'num_times_chosen_since_improvement': cell_data.get('num_times_chosen_since_improvement', 0),
#     }
    
#     for attr_name, attr_value in attributes.items():
#         wa = CELL_SCORE_WEIGHTS[attr_name]
#         pa = CELL_SCORE_POWERS[attr_name]
#         cnt_score = wa * (1.0 / ((attr_value + EPSILON_1) ** pa)) + EPSILON_2
#         score += cnt_score
    
#     return score

# def select_cell_from_archive(archive: Dict) -> tuple:
#     """Select a cell from archive using probability distribution based on scores.
    
#     Returns: (selected_hash, obs, state, best_reward) or (None, None, None, 0) if archive is empty
#     """
#     if not archive:
#         return None, None, None, 0
    
#     # Calculate scores for all cells
#     cell_hashes = list(archive.keys())
#     scores = np.array([calculate_cell_score(archive[h]) for h in cell_hashes])
    
#     # Normalize to get probabilities
#     probabilities = scores / scores.sum()
    
#     # Select a cell based on probabilities
#     selected_idx = np.random.choice(len(cell_hashes), p=probabilities)
#     selected_hash = cell_hashes[selected_idx]
    
#     # Update the selected cell's count
#     archive[selected_hash]['num_times_chosen'] += 1
    
#     return selected_hash, archive[selected_hash]['obs'], archive[selected_hash]['state'], archive[selected_hash].get('best_reward_seen', 0)

# def track_archive_improvement(archive: Dict, prev_archive_size: int):
#     """Track improvement by resetting counts for cells that discovered new cells."""
#     if len(archive) > prev_archive_size:
#         # New cells were discovered, reset improvement counter for recently chosen cells
#         for h in archive:
#             archive[h]['num_times_chosen_since_improvement'] = 0

# def reset_to_cell(parcel: Dict, env: gym.Env, state, cell_obs: np.ndarray):
#     """Reset environment to an archived cell by restoring its state."""
#     if state is not None:
#         env.unwrapped.restore_state(state)
#         # Set the observation to the archived one (should match the restored state)
#         parcel['obs'] = cell_obs.copy()
#         parcel['_reset_to_cell'] = True

# def go_explore(env, total_timesteps: int):
#     """Implement Go-Explore algorithm with cell resetting (the full loop)."""
#     archive = {}  # hash -> {'obs': np.array, 'num_times_chosen': int, 'num_times_visited': int, ...}
#     trajectories = []  # List of trajectories for BC
#     current_trajectory = []

#     # Exploration phase: Use random actions to discover states
#     exploration_steps = 3 * total_timesteps // 4
#     collector = tp_utils.Collector(['obs', 'action', 'reward'])
    
#     def get_train_participants_exploration():
#         with tp_utils.StepsTracker(exploration_steps, desc="exploration steps") as steps_tracker:
#             yield functools.partial(tp_gym_utils.call_reset, env=env)
#             yield from itertools.cycle([
#                 get_random_action(num_actions=env.action_space.n),
#                 functools.partial(tp_gym_utils.call_step, env=env),
#                 collector,
#                 functools.partial(archive_state, archive=archive, env=env),
#                 functools.partial(collect_trajectory_for_bc, trajectories=trajectories, archive=archive, current_trajectory=current_trajectory),
#                 steps_tracker,
#                 functools.partial(call_reset_done_single, env=env),
#             ])

#     exploration_assembly = tp.Assembly(get_train_participants_exploration)
#     exploration_assembly.launch()

#     print(f"Archived {len(archive)} states during exploration.")
#     max_reward_found = max([archive[h].get('best_reward_seen', 0) for h in archive], default=0)
#     print(f"Max reward found during exploration: {max_reward_found}")

#     # Robustification phase: Train policy using BC on collected trajectories
#     policy = StateToActionLogits(env.observation_space.shape[0], env.action_space.n)
#     policy = policy.to(device)
#     optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

#     train_bc_policy(trajectories, policy, optimizer)

#     print("Trained policy using Behavioral Cloning.")

#     # Exploitation phase: Use cell selection + resetting + policy (the full Go-Explore loop)
#     # Key difference: periodically select a cell from archive, reset to it, then explore from there
#     exploitation_steps = total_timesteps - exploration_steps
#     trajectories.clear()
#     current_trajectory.clear()

#     cell_reset_counter = 0
#     cell_reset_frequency = 100  # Reset to a new cell every N steps

#     def cell_selector(parcel: Dict):
#         """Periodically select and reset to cells from the archive."""
#         nonlocal cell_reset_counter, env
#         cell_reset_counter += 1
#         if cell_reset_counter % cell_reset_frequency == 0 and archive:
#             selected_hash, cell_obs, cell_state, cell_reward = select_cell_from_archive(archive)
#             reset_to_cell(parcel, env, cell_state, cell_obs)

#     def get_train_participants_exploitation():
#         with tp_utils.StepsTracker(exploitation_steps, desc="exploitation steps") as steps_tracker:
#             yield functools.partial(tp_gym_utils.call_reset, env=env)
#             yield from itertools.cycle([
#                 cell_selector,  # Periodically select and reset to cells (the "Go" in Go-Explore)
#                 functools.partial(get_action_policy, agent=policy),
#                 functools.partial(tp_gym_utils.call_step, env=env),
#                 collector,
#                 functools.partial(archive_state, archive=archive, env=env),
#                 steps_tracker,
#                 functools.partial(call_reset_done_single, env=env),
#             ])

#     exploitation_assembly = tp.Assembly(get_train_participants_exploitation)
#     exploitation_assembly.launch()

#     print(f"Total archived states: {len(archive)}")
#     max_reward_found = max([archive[h].get('best_reward_seen', 0) for h in archive], default=0)
#     print(f"Max reward found: {max_reward_found}")

#     return policy, archive


def evaluate(env, agent, num_episodes: int) -> float:
    agent.eval()

    rewards_collector = tp_utils.Collector(['reward'])

    def get_participants():
        yield functools.partial(tp_gym_utils.call_reset, env=env)
        yield from itertools.cycle([
            functools.partial(get_action_policy, agent=agent),
            functools.partial(tp_gym_utils.call_step, env=env),
            rewards_collector,
            tp_gym_utils.check_done
        ])

    evaluate_assembly = tp.Assembly(get_participants)

    for _ in range(num_episodes):
        _ = evaluate_assembly.launch()
        # Note that we don't clear the rewards in 'rewards_collector'), and so we continue to collect.

    total_reward = sum(x['reward'] for x in rewards_collector.get_entries())

    return total_reward / num_episodes


def clone_state(parcel: Dict, env):
    parcel['state'] = env.unwrapped.clone_state()


def train(env, total_timesteps: int):

    collector = tp_utils.Collector(['state', 'obs', 'action', 'reward', 'new_obs'])
    exploration_steps = total_timesteps

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

    class VisitedState:
        times_visited: int = 0
        times_selected: int = 0
        total_reward_collected: float = 0
        state: Any

        def __str__(self):
            return f'times_visited={self.times_visited} total_reward_collected={self.total_reward_collected}'

    seen_obs = defaultdict(VisitedState)

    def update_exploration_memory(obs, action, new_obs, state, reward):
        new_obs_downsampled = downsample_obs(new_obs)
        key = hashlib.sha256(new_obs_downsampled.tobytes()).hexdigest()
        entry = seen_obs[key]
        entry.times_visited += 1
        entry.state = state # potentially override but okay
        entry.total_reward_collected += reward

    def score(entry, total_selections):
        return math.sqrt(2 * math.log(total_selections + 1) / max(1, entry.times_selected))

    total_selections: int = 0

    def explore_from_cell(entry):
        # Restore the saved emulator state
        env.unwrapped.restore_state(entry.state)

        # Try each action once
        for action in range(env.action_space.n):
            env.unwrapped.restore_state(entry.state)  # reset before each action
            obs, reward, done, truncated, info = env.step(action)
            update_exploration_memory(obs, action, obs, env.unwrapped.clone_state(), reward)

    def select_state_and_go_there():
        nonlocal total_selections
        total_selections += 1
        entry = max(seen_obs.values(), key=lambda e: score(e, total_selections))
        entry.times_selected += 1
        # print(f'select_state_and_go_there -> {entry}')
        explore_from_cell(entry)
        env.unwrapped.restore_state(entry.state)

    def check_if_time_to_reset(parcel: Dict):
        if not (parcel.get('terminated', False) or parcel.get('truncated', False) or parcel.get('step', 0) % 200 == 0):
            return
        for entry in collector.get_entries():
            obs = entry['obs']
            action = entry['action']
            new_obs = entry['new_obs']
            state = entry['state']
            reward = entry['reward']
            if reward > 0:
                print(f'I see a reward in my hard work: {reward}')
            update_exploration_memory(obs, action, new_obs, state, reward)
        collector.clear_entries()
        select_state_and_go_there()

    def advance(parcel: Dict):
        parcel['obs'] = parcel['new_obs']

    def ram_debug(parcel: Dict, env):
        ram = env.unwrapped.ale.getRAM()
        print(f'({ram[42]}, {ram[43]})') # x, y of the player

    def get_participants_exploration():
        with tp_utils.StepsTracker(exploration_steps, desc="exploration steps") as steps_tracker:
            yield functools.partial(tp_gym_utils.call_reset, env=env)
            yield from itertools.cycle([
                # functools.partial(ram_debug, env=env),
                functools.partial(clone_state, env=env),
                functools.partial(get_random_action, env=env),
                functools.partial(tp_gym_utils.call_step, env=env, save_obs_as='new_obs'),
                collector,
                advance,
                steps_tracker,
                check_if_time_to_reset
            ])

    exploration_assembly = tp.Assembly(get_participants_exploration)
    _ = exploration_assembly.launch()
    print(f'len for seen_obs={len(seen_obs)}')


def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    gym.register_envs(ale_py)

    env = make_env(seed=0)

    total_timesteps = 200_000  # Adjust as needed

    train(env, total_timesteps)

    # policy, archive = go_explore(env, total_timesteps)



    # # Evaluate the final policy
    # mean_reward = evaluate(env, policy, 10)
    # print(f"Final mean reward: {mean_reward}")

    # env.close()

    # # Record video of the trained agent
    # rendering_env = make_env(seed=1, render_mode="rgb_array_list", max_episode_steps=1_000)
    # env_name = gym_environment.split("/")[-1]
    # video_name = f"{env_name}-go-explore-trained.mp4"
    # record_video(rendering_env, policy, videos_path / video_name)
    # rendering_env.close()
    # print("Video recorded.")


if __name__ == "__main__":
    main()