import datetime
import functools
import itertools
import random
from typing import Dict, List
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint.tensorboard_utils as tp_tb_utils
import turingpoint.pandas_utils as tp_pd_utils
import turingpoint as tp


class StateToQValues(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_features = 64
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.Tanh(), # ReLU(),
            # nn.Linear(in_features=hidden_features, out_features=hidden_features),
            # nn.Tanh(), # ReLU(),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )
        self.out_features = out_features # keep for the range of actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> q-values (regression)"""

        return self.net(obs)
    

def get_action(parcel: Dict, *, agent: StateToQValues, explore=False):
    """'participant' representing the agent. Epsion greedy strategy:
    1 - epsion - take the "best action" - explotation
    epsilon - take a random action - exploration (can still by chance take the "best action")
    """

    if explore and np.random.binomial(1, parcel['epsilon']) > 0:
        parcel['action'] = np.random.choice(range(agent.out_features))
    else:
        obs = parcel['obs']
        q_values = agent(torch.tensor(obs))
        _, action = q_values.max(dim=0) # or just argmax..
        parcel['action'] = action.item()


def evaluate(env, agent, num_episodes: int) -> float:
    """Collect episodes and calculate the mean total reward."""

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

    for _ in trange(num_episodes, desc="evaluate"):
        _ = evaluate_assembly.launch()
        # Note that we don't clear the rewards in 'rewards_collector', and so we continue to collect.

    total_reward = sum(x['reward'] for x in rewards_collector.get_entries())

    return total_reward / num_episodes


def train(env, agent, target_critic, total_timesteps):
    """Given a model (agent) and a critic
    (which should be of the same sort and which values are overridden in the begining).
    Train the model"""

    discount = 0.99
    K = 4
    batch_size = 128

    replay_buffer_collector = tp_pd_utils.ReplayBufferCollector(
        collect=['obs', 'action', 'reward', 'terminated', 'truncated', 'next_obs'])

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-2)
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    max_epsilon = 0.15
    min_epsilon = 0.05

    def update_target_critic_from_agent(target_critic, agent):
        target_critic.load_state_dict(agent.state_dict())

    def learn(parcel: dict):

        if parcel['step'] == 0:
            parcel['lr'] = optimizer.param_groups[0]['lr']

        if parcel['step'] % 1000 == 0:
            update_target_critic_from_agent(target_critic, agent)

        if parcel['step'] < 200: # we'll start really learning only after we collect some steps
            return

        replay_buffer = replay_buffer_collector.replay_buffer

        parcel['Mean Rewards/train'] = replay_buffer['reward'].mean() # taking from the replay_buffer ? TODO: !!!

        losses = []

        for _ in range(K):

            batch = replay_buffer.sample(min(batch_size, len(replay_buffer)), replace=False) # random_state? to use from the iter round?

            # Now learn from the batch

            obs = torch.tensor(np.array(batch['obs'].tolist()), dtype=torch.float32)
            action = torch.tensor(batch['action'].to_list())
            reward = torch.tensor(batch['reward'].values, dtype=torch.float32)
            next_obs = torch.tensor(np.array(batch['next_obs'].to_list()), dtype=torch.float32)
            terminated = torch.tensor(batch['terminated'].to_list())
            # truncated = torch.tensor(batch['truncated'].to_list())

            with torch.no_grad():
                next_obs_q_values = agent(next_obs)
                _, next_obs_q_value_idx = next_obs_q_values.max(dim=1)
                next_obs_q_value = torch.gather(
                    target_critic(next_obs),
                    dim=1,
                    index=next_obs_q_value_idx.view(-1, 1)
                ).squeeze()
                target = reward + torch.where(terminated, 0, discount * next_obs_q_value)

            q_value = torch.gather(agent(obs), dim=1, index=action.view(-1, 1)).squeeze()

            loss = F.mse_loss(q_value, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            if parcel['step'] % 5000 == 0:
                scheduler.step()
                parcel['lr'] = optimizer.param_groups[0]['lr'] # scheduler.get_last_lr()

            losses.append(loss.item())

        parcel['Loss/train'] = np.mean(losses)

    def set_epsilon(parcel: dict):
        parcel['epsilon'] = (
            min_epsilon
            + (max_epsilon - min_epsilon) * ((total_timesteps - parcel['step']) / total_timesteps)
        )

    def advance(parcel: dict):
        parcel['obs'] = parcel.pop('next_obs')

    def reset_if_needed(parcel: dict):
        terminated = parcel.pop('terminated', False)
        truncated = parcel.pop('truncated', False) 
        if terminated or truncated:
            tp_gym_utils.call_reset(parcel, env=env)

    def get_train_participants():
        with (tp_tb_utils.Logging(
            path=f"runs/LunarLander_ddqn_{datetime.datetime.now().strftime('%I_%M%p_on_%B_%d_%Y')}_{discount=}",
            track=[
                'Mean Rewards/train',
                'Loss/train',
            ]) as logging,
            tp_utils.StepsTracker(total_timesteps=total_timesteps, desc="training steps") as steps_tracker):

            yield functools.partial(tp_gym_utils.call_reset, env=env)
            yield steps_tracker # initialization to 0
            yield from itertools.cycle([
                set_epsilon,
                functools.partial(get_action, agent=agent, explore=True),
                functools.partial(tp_gym_utils.call_step, env=env, save_obs_as="next_obs"),
                replay_buffer_collector,
                learn,
                logging,
                steps_tracker, # can raise Done
                advance,
                reset_if_needed
            ])

    train_assembly = tp.Assembly(get_train_participants)
    
    train_assembly.launch()


def main():

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    env = gym.make('LunarLander-v3')

    env.reset(seed=1)

    # state and obs/observations are used in this example interchangably.

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = StateToQValues(state_space, action_space)
    target_critic = StateToQValues(state_space, action_space)

    mean_reward_before_train = evaluate(env, agent, 100)
    print("before training")
    print(f'{mean_reward_before_train=}')

    train(env, agent, target_critic, total_timesteps=1_000_000)

    mean_reward_after_train = evaluate(env, agent, 100)
    print("after training")
    print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
    main()
