import datetime
import functools
import itertools
import random
from typing import Dict, List
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import ale_py
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint as tp


class StateToQValues(nn.Module):
    """Copied mostly from Berkeley CS 285/homework_fall2023
    """
    def __init__(self, out_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, out_features),
        )
        self.out_features = out_features # keep for the range of actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> q-values (regression)"""
        assert obs.ndim == 4, f'{obs.shape=}'
        # nn.Flatten() from above skips first dim, so if no batch, we fail to flatten all needed.

        return self.net(obs)


def get_action(parcel: Dict, *, agent: StateToQValues, epsilon=None):
    """'participant' representing the agent. Epsion greedy strategy:
    1 - epsion - take the "best action" - explotation
    epsilon - take a random action - exploration (can still by chance take the "best action")
    """

    if epsilon is not None and np.random.binomial(1, epsilon) > 0:
        parcel['action'] = np.random.choice(range(agent.out_features))
    else:
        obs = parcel['obs']
        q_values = agent(torch.tensor(obs).unsqueeze(dim=0))
        _, action = q_values.squeeze().max(dim=0) # or just argmax..
        parcel['action'] = action.item()


def evaluate(env, agent, num_episodes: int) -> float:

    rewards_collector = tp_utils.Collector(['reward'])

    def get_participants():
        yield functools.partial(tp_gym_utils.call_reset, env=env)
        yield from itertools.cycle([
                functools.partial(get_action, agent=agent),
                functools.partial(tp_gym_utils.call_step, env=env),
                rewards_collector,
                tp_gym_utils.check_done
        ])

    evaluate_assembly = tp.Assembly(get_participants)

    for _ in range(num_episodes):
        _ = evaluate_assembly.launch()
        # Note that we don't clear the rewards in 'rewards_collector', and so we continue to collect.

    total_reward = sum(x['reward'] for x in rewards_collector.get_entries())

    return total_reward / num_episodes


def collect_episodes(env, agent, num_episodes=40, epsilon=0.1) -> List[Dict]:

    collector = tp_utils.Collector(['obs', 'action', 'reward', 'terminated', 'truncated', 'next_obs'])

    def advance(parcel: dict):
        parcel['obs'] = parcel.pop('next_obs')

    def get_episode_participants():
        yield functools.partial(tp_gym_utils.call_reset, env=env)
        yield from itertools.cycle([
                functools.partial(get_action, agent=agent, epsilon=epsilon),
                functools.partial(tp_gym_utils.call_step, env=env, save_obs_as="next_obs"),
                collector,
                tp_gym_utils.check_done,
                advance
        ])

    episodes_assembly = tp.Assembly(get_episode_participants)

    steps = [] # it will be a list of dictionaries
    rewards = []
    for _ in range(num_episodes):
        _ = episodes_assembly.launch()
        episode = list(collector.get_entries())
        episode_reward = sum(x['reward'] for x in episode) # TODO: discount factor?
        rewards.append(episode_reward)
        collector.clear_entries()
        steps.extend(episode)
    return steps, np.mean(rewards)


def train(env, agent, target_critic, total_timesteps):
    discount = 0.99

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-2)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    writer = SummaryWriter(
        f"runs/MsPacman_ddqn_{datetime.datetime.now().strftime('%I_%M%p_on_%B_%d_%Y')}_{discount=}"
    ) # TensorBoard

    replay_buffer = None

    K = 2

    max_epsilon = 0.15
    min_epsion = 0.05

    timesteps = 0
    updates = 0
    with tqdm(total=total_timesteps, desc="training steps") as pbar:
        while timesteps < total_timesteps:

            epsilon = (
                min_epsion
                + (max_epsilon - min_epsion) * ((total_timesteps - timesteps) / total_timesteps)
            )

            episodes, mean_reward = collect_episodes(env, agent, num_episodes=3, epsilon=epsilon)
            writer.add_scalar("Mean Rewards/train", mean_reward, timesteps)

            new_entries = pd.DataFrame.from_records(episodes)
            replay_buffer = (
                pd.concat([replay_buffer, new_entries]) # it is okay to have None. None is dropped silentely..
                .tail(10_000) # keep latest, drop oldest
            )

            for _ in range(K):

                batch = replay_buffer.sample(min(128, len(replay_buffer)), replace=False) # random_state? to use from the iter round?

                # Now learn from the batch

                obs = torch.tensor(np.array(batch['obs'].tolist()), dtype=torch.float32)
                action = torch.tensor(batch['action'].to_list())
                reward = torch.tensor(batch['reward'].values, dtype=torch.float32)
                next_obs = torch.tensor(np.array(batch['next_obs'].to_list()))
                terminated = torch.tensor(batch['terminated'].to_list())
                # truncated = torch.tensor(batch['truncated'].to_list())

                with torch.no_grad():
                    next_obs_q_values = agent(next_obs)
                    _, next_obs_q_value_idx = next_obs_q_values.max(dim=1)
                    next_obs_q_value = torch.gather(
                        target_critic(obs),
                        dim=1,
                        index=next_obs_q_value_idx.view(-1, 1)
                    ).squeeze()
                    target = reward + torch.where(terminated, 0, discount * next_obs_q_value)

                q_value = torch.gather(agent(obs), dim=1, index=action.view(-1, 1)).squeeze()

                loss = F.mse_loss(q_value, target)

                optimizer.zero_grad()

                timesteps += len(batch)

                writer.add_scalar("Loss/train", loss, timesteps)

                loss.backward()

                optimizer.step()

                updates += 1
                if updates % 1000 == 0:
                    update_target_critic_from_agent(target_critic, agent)

                pbar.update(len(batch))

            scheduler.step()

    writer.flush()


def update_target_critic_from_agent(target_critic, agent):
     target_critic.load_state_dict(agent.state_dict())


def main():

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    gym.register_envs(ale_py)

    env = gym.make("ALE/MsPacman-v5", frameskip=1)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=3,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=True,
    )
    env = FrameStackObservation(env, stack_size=4)

    env.reset(seed=1)

    # state and obs/observations are used in this example interchangably.

    # state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = StateToQValues(action_space)
    target_critic = StateToQValues(action_space)
    update_target_critic_from_agent(target_critic, agent) # They should start from the same values for the parameters

    mean_reward_before_train = evaluate(env, agent, 100)
    print("before training")
    print(f'{mean_reward_before_train=}')

    train(env, agent, target_critic, total_timesteps=3_000) # 1_000_000)

    mean_reward_after_train = evaluate(env, agent, 100)
    print("after training")
    print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
    main()
