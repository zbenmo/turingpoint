import datetime
import functools
import itertools
import random
from typing import Dict
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange
from contextlib import contextmanager

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint.tensorboard_utils as tp_tb_utils
import turingpoint.torch_utils as tp_torch_utils
import turingpoint as tp


class StateToAction(nn.Module):
    def __init__(self, in_features, out_actions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_features = 64
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            # nn.Dropout(p = 0.2),
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.Tanh(), # ReLU(),
            # nn.Dropout(p = 0.2),
            # nn.Linear(in_features=hidden_features, out_features=hidden_features),
            # nn.BatchNorm1d(hidden_features),
            # nn.Tanh(), # ReLU(),
            nn.Linear(in_features=hidden_features, out_features=out_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> action (regression)"""

        return self.net(obs)


class StateActionToQValue(nn.Module):
    def __init__(self, in_features, in_actions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_features = 64
        self.net = nn.Sequential(
            nn.BatchNorm1d((in_features + in_actions)),
            # nn.Dropout(p = 0.2),
            nn.Linear(in_features=(in_features + in_actions), out_features=hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.Tanh(), # ReLU(),
            # nn.Dropout(p = 0.2),
            # nn.Linear(in_features=hidden_features, out_features=hidden_features),
            # nn.BatchNorm1d(hidden_features),
            # nn.Tanh(), # ReLU(),
            nn.Linear(in_features=hidden_features, out_features=1),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """obs, action -> q-values (regression)"""

        return self.net(torch.concat((obs, actions), dim=1))


def get_action(parcel: Dict, *, agent: StateToAction, explore=False):
    """'participant' representing the agent. Epsion greedy strategy:
    1 - epsion - take the "best action" - explotation
    epsilon - take a random action - exploration (can still by chance take the "best action")
    """

    # if explore and np.random.binomial(1, parcel['epsilon']) > 0:
    #     parcel['action'] = np.random.choice(range(agent.out_features))
    # else:
    #     obs = parcel['obs']
    #     q_values = agent(torch.tensor(obs))
    #     _, action = q_values.max(dim=0) # or just argmax..
    #     parcel['action'] = action.item()
    obs = parcel['obs']
    # print(f'{obs.shape=}')
    assert not agent.training
    action = agent(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
    if explore:
        action = torch.clip(action + torch.rand_like(action), -0.4, 0.4) # -2.0, 2.0) # for Pendulum-v1
    parcel['action'] = action.squeeze().detach().numpy() # .item()
    if len(parcel['action'].shape) == 0:
        parcel['action'] = parcel['action'].reshape((1, )) # for Pendulum-v1


def evaluate(env, agent, num_episodes: int) -> float:
    """Collect episodes and calculate the mean total reward."""

    agent.eval()

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



@contextmanager
def start_train(agent: StateToAction):
    try:
        agent.train()
        yield agent
    finally:
        agent.eval()


def train(env, agent: StateToAction, critic: StateActionToQValue, total_timesteps):
    """Given a model (agent) and a critic
    (which should be of the same sort and which values are overridden in the begining).
    Train the model"""

    state_space = env.observation_space.shape[0] # alternatively take it from agent/target...
    action_space = env.action_space.shape[-1]
    target_agent = StateToAction(state_space, out_actions=action_space)
    target_critic = StateActionToQValue(state_space, action_space)
    target_agent.load_state_dict(agent.state_dict()) # initialize the policy and the target with the same (random) values
    target_critic.load_state_dict(critic.state_dict()) # same here

    agent.eval() # when we'll actually train, we'll say it explicitly below (in learn)
    critic.eval() # same here

    target_agent.eval()
    target_critic.eval()

    discount = 0.99
    K = 1
    batch_size = 128

    replay_buffer_collector = tp_utils.ReplayBufferCollector(
        collect=['obs', 'action', 'reward', 'terminated', 'truncated', 'next_obs'])

    optimizer_agent = torch.optim.Adam(agent.parameters(), lr=1e-5)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-5)
    # scheduler_agent = ExponentialLR(optimizer_agent, gamma=0.99)
    # scheduler_critic = ExponentialLR(optimizer_critic, gamma=0.99)

    # max_epsilon = 0.15
    # min_epsilon = 0.05

    def update_target(target_agent, target_critic, agent, critic):
        tau = 0.05 # if needed, can use two different values
        tp_torch_utils.polyak_update(agent.parameters(), target_agent.parameters(), tau) # agent -> target_agent
        tp_torch_utils.polyak_update(critic.parameters(), target_critic.parameters(), tau) # critic -> target_critic

    def learn(parcel: dict):

        with start_train(agent), start_train(critic):

            if parcel['step'] == 0:
                parcel['lr_agent'] = optimizer_agent.param_groups[0]['lr']
                parcel['lr_critic'] = optimizer_critic.param_groups[0]['lr']

            if parcel['step'] % 1 == 0:
                update_target(target_agent, target_critic, agent, critic)

            if parcel['step'] < 200: # we'll start really learning only after we collect some steps
                return

            replay_buffer = replay_buffer_collector.replay_buffer

            rewards = (x['reward'] for x in replay_buffer)
            parcel['Mean Rewards/train'] = sum(rewards) / len(replay_buffer) # taking from the replay_buffer ? TODO: !!!

            losses_agent = []
            losses_critic = []

            replay_buffer_dataloader = torch.utils.data.DataLoader(
                replay_buffer, batch_size=batch_size, shuffle=True)

            for _, batch in zip(range(K), replay_buffer_dataloader):

                if len(batch['obs']) < 2:
                    continue

                # Now learn from the batch

                obs = batch['obs'].to(torch.float32)
                action = batch['action']
                reward = batch['reward'].to(torch.float32)
                next_obs = batch['next_obs'].to(torch.float32)
                terminated = batch['terminated']
                # truncated = batch['truncated']

                with torch.no_grad():
                    next_actions = target_agent(next_obs)
                    next_obs_q_values = target_critic(next_obs, next_actions).squeeze()
                    target = reward + torch.where(terminated, 0, discount * next_obs_q_values)

                optimizer_critic.zero_grad()

                q_value = critic(obs, action).squeeze()

                loss = F.mse_loss(q_value, target)

                loss.backward()

                losses_critic.append(loss.item())

                optimizer_critic.step()

                optimizer_agent.zero_grad()

                loss = -critic(obs, agent(obs)).mean() # let's maximize this value (hence the minus sign)

                loss.backward()

                optimizer_agent.step()

                losses_agent.append(loss.item())

                # if parcel['step'] % 5000 == 0:
                #     scheduler_critic.step()
                #     scheduler_agent.step()
                #     parcel['lr_critic'] = optimizer_critic.param_groups[0]['lr'] # scheduler.get_last_lr()
                #     parcel['lr_agent'] = optimizer_agent.param_groups[0]['lr'] # scheduler.get_last_lr()

            parcel['Loss(agent)/train'] = np.mean(losses_agent)
            # losses_agent_std = np.std(losses_agent)
            # parcel['Loss(agent)+std/train'] = parcel['Loss(agent)/train'] + losses_agent_std
            # parcel['Loss(agent)-std/train'] = parcel['Loss(agent)/train'] - losses_agent_std
            parcel['Loss(critic)/train'] = np.mean(losses_critic)
            # losses_critic_std = np.std(losses_critic)
            # parcel['Loss(critic)+std/train'] = parcel['Loss(critic)/train'] + losses_critic_std
            # parcel['Loss(critic)-std/train'] = parcel['Loss(critic)/train'] - losses_critic_std

    # def set_epsilon(parcel: dict):
    #     parcel['epsilon'] = (
    #         min_epsilon
    #         + (max_epsilon - min_epsilon) * ((total_timesteps - parcel['step']) / total_timesteps)
    #     )

    def advance(parcel: dict):
        parcel['obs'] = parcel.pop('next_obs')

    def reset_if_needed(parcel: dict):
        terminated = parcel.pop('terminated', False)
        truncated = parcel.pop('truncated', False) 
        if terminated or truncated:
            tp_gym_utils.call_reset(parcel, env=env)

    # def take_interesting_info(parcel: dict):
    #     parcel.update(parcel.pop('info'))

    def get_train_participants():
        with (tp_tb_utils.Logging(
            path=f"runs/Humanoid_ddpg_{datetime.datetime.now().strftime('%H_%M%p_on_%B_%d_%Y')}_{discount=}",
            track=[
                'Mean Rewards/train',
                'Loss(agent)/train',
                'Loss(critic)/train',
                'lr_critic',
                'lr_agent',

                # 'reward_survive',
                # 'reward_forward',

                # 'Loss(agent)+std/train',
                # 'Loss(agent)-std/train',
                # 'Loss(critic)+std/train',
                # 'Loss(critic)-std/train',
            ]) as logging,
            tp_utils.StepsTracker(total_timesteps=total_timesteps, desc="training steps") as steps_tracker):

            yield functools.partial(tp_gym_utils.call_reset, env=env)
            yield steps_tracker # initialization to 0
            yield from itertools.cycle([
                # set_epsilon,
                functools.partial(get_action, agent=agent, explore=True),
                functools.partial(tp_gym_utils.call_step, env=env, save_obs_as="next_obs"),
                replay_buffer_collector,
                learn,
                # take_interesting_info, # those will be also logged.
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

    env = gym.make('Humanoid-v5') #  "Pendulum-v1") #  # gym.make('Humanoid-v5')

    env.reset(seed=1)

    # state and obs/observations are used in this example interchangably.

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[-1]

    # print(f'{state_space=}')
    # print(f'{action_space=}')

    act = StateToAction(state_space, out_actions=action_space) # This is the agent
    critic = StateActionToQValue(state_space, action_space)

    mean_reward_before_train = evaluate(env, act, 100)
    print("before training")
    print(f'{mean_reward_before_train=}')

    train(env, act, critic, total_timesteps=20_000)

    mean_reward_after_train = evaluate(env, act, 100)
    print("after training")
    print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
    main()
