import random
from typing import Dict, List, Tuple
import gymnasium as gym
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import (
    DQNPolicy
)
import torch
from torch import nn

from turingpoint.gymnasium_utils import (
  EnvironmentParticipant,
  GymnasiumAssembly,
)
from turingpoint.utils import (
  Collector,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def get_agent(env, device="cpu") -> Tuple[DQNPolicy, torch.optim.Optimizer]:
  class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

  state_shape = env.observation_space.shape or env.observation_space.n
  action_shape = env.action_space.shape or env.action_space.n
  net = Net(state_shape, action_shape)
  optim = torch.optim.Adam(net.parameters(), lr=1e-3)
  policy = DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=4)
  return policy, optim


def main():
  RANDOM_SEED=1
  random.seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)
  torch.manual_seed(RANDOM_SEED)

  env = gym.make('CartPole-v1')
  env.reset(seed=RANDOM_SEED)
  # env.action_space.seed(RANDOM_SEED)
  agent, optim = get_agent(env)

  def agent_participant(parcel: dict):
    action = None # till shown otherwise
    # may take a random action as to explore
    if np.random.binomial(1, agent.eps):
      action = np.random.choice([0, 1])
    else:
      input_batch = Batch(
        obs=np.array([parcel['obs']]),
        info=np.array([{}])
      )
      output_batch = agent(input_batch, None)
      action = output_batch.act[0]
    parcel['action'] = action

  def evaluate(num_episodes: int) -> Tuple[float, List[float]]:
    """
    Args:
      num_episodes: the evaluation is done for this number of episodes.

    Returns:
      The mean of the rewards and the rewards as a list.
    """

    agent.set_eps(0.00)

    # Note: Tianshou also has a concept of a Collector.
    # Below is a Turingpoint Collector which is just yet another participant.
    rewards_collector = Collector(['reward'])

    assembly = GymnasiumAssembly(env, [
      # print_parcel,
      agent_participant,
      EnvironmentParticipant(env),
      rewards_collector
    ])

    rewards = []

    for _ in range(num_episodes):
      _ = assembly.launch()
      total_reward = sum(x['reward'] for x in rewards_collector.get_entries())
      rewards.append(total_reward)
      rewards_collector.clear_entries()

    return np.mean(rewards), rewards 

  def to_tianshou_buffer(collection: List[Dict]) -> ReplayBuffer:
    rb = ReplayBuffer(len(collection))
    for entry in collection:
      rb.add(Batch(
        obs=entry['obs'],
        act=entry['action'],
        rew=entry['reward'],
        terminated=entry['terminated'],
        truncated=entry['truncated'],
        obs_next=entry['obs_next']
      ))
    return rb

  def train(total_timesteps):

    # Note: Tianshou also has a concept of a Collector.
    # Below is a Turingpoint Collector which is just yet another participant.
    collector = Collector(['obs', 'action', 'reward', 'terminated', 'truncated', 'obs_next'])

    steps = 0

    def end_iteration(parcel: dict):
      nonlocal steps

      steps += 1
      parcel['obs'] = parcel['obs_next']
      del parcel['obs_next']

    assembly = GymnasiumAssembly(env, [
      agent_participant,
      EnvironmentParticipant(env, save_obs_as='obs_next'),
      collector,
      end_iteration
    ])

    agent.set_eps(0.10)

    losses = []
    steps_record = []
    last_steps = 0
    with tqdm(total=total_timesteps, desc='steps') as pb:
      while steps <= total_timesteps:
        parcel = assembly.launch() # one episode

        pb.update(steps - last_steps)
        last_steps = steps
        replay_buffer = to_tianshou_buffer(list(collector.get_entries()))
        ret_from_update = agent.update(0, replay_buffer) 
        losses.append(ret_from_update['loss'])
        steps_record.append(steps)
        # should I clear the entries? should I clear first 64 entries? 64 stands for batch size? TODO: ..
        collector.clear_entries()

    return steps_record, losses

  mean_reward_before_train, _ = evaluate(100)
  print("before training")
  print(f'{mean_reward_before_train=}')

  steps_record, losses = train(total_timesteps=10_000)

  mean_reward_after_train, rewards = evaluate(100)
  print("after training")
  print(f'{mean_reward_after_train=}')

  plt.plot(steps_record, losses)
  plt.title("losses during training")
  plt.ylim(0, 20)
  plt.show()

  plt.plot(rewards)
  plt.ylim(0, 220)
  plt.title("rewards")
  plt.show()


if __name__ == "__main__":
  main()