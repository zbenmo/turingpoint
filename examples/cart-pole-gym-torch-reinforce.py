import functools
import itertools
import random
from typing import Dict, List
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint as tp


class StateToActionLogProb(nn.Module):
  def __init__(self, in_features, out_features, *args, **kwargs):
    super().__init__(*args, **kwargs)
    hidden_features = 10
    self.net = nn.Sequential(
      nn.Linear(in_features=in_features, out_features=hidden_features),
      nn.ReLU(),
      nn.Linear(in_features=hidden_features, out_features=out_features),
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """obs -> actions (log_probabilities)"""

    return self.net(obs)


def get_action(parcel: Dict, *, agent: StateToActionLogProb):
  """Picks a random action based on the probabilities that the agent assigns.
  Just needs to account for the fact the the agent actually returns log probabilities rather than probabilities.
  """
  obs = parcel['obs']
  log_probs = agent(torch.tensor(obs))
  probs = torch.exp(log_probs)
  probs /= probs.sum()
  action = torch.multinomial(probs, 1).item()
  parcel['action'] = action
  parcel['log_prob'] = log_probs[action] # may be useful for the training (note: still a tensor)


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


def collect_episodes(env, agent, num_episodes=40) -> List[List[Dict]]:

  collector = tp_utils.Collector(['log_prob', 'reward']) # it seems that those are enough for Reinforce

  def get_episode_participants():
    yield functools.partial(tp_gym_utils.call_reset, env=env)
    yield from itertools.cycle([
        functools.partial(get_action, agent=agent),
        functools.partial(tp_gym_utils.call_step, env=env),
        collector,
        tp_gym_utils.check_done
    ])

  episodes_assembly = tp.Assembly(get_episode_participants)

  episodes = [] # it will be a list of lists of dictionaries
  for _ in range(num_episodes):
    _ = episodes_assembly.launch()
    episode = list(collector.get_entries())
    collector.clear_entries()
    episodes.append(episode)
  return episodes


def train(env, agent, total_timesteps):
  causality_to_be_accounted_for = True
  normilize_the_rewards = True
    
  optimizer = torch.optim.Adam(agent.parameters(), lr=0.0001)
  writer = SummaryWriter(f"runs/reinforce_{causality_to_be_accounted_for=}_{normilize_the_rewards=}") # TensorBoard

  timesteps = 0
  with tqdm(total=total_timesteps, desc="training steps") as pbar:
    while timesteps < total_timesteps:
      episodes = collect_episodes(env, agent)

      # Now learn from above episodes

      # disclaimer: a lot more of vectorization can be done here.
      # potentially a usage of pandas/polars, numpy, or torch. 
      # I'm KISSing it here.

      log_probs_batch = []
      rewards_batch = []

      total_rewards = []

      for episode in episodes:
        log_prob, rewards = zip(*((e['log_prob'], e['reward']) for e in episode))
        total_reward = sum(rewards)
        total_rewards.append(total_reward)
        log_probs_batch.extend(log_prob)
        if causality_to_be_accounted_for:
          rewards_batch.extend(tp_utils.discounted_reward_to_go(rewards))
        else:
          rewards_batch.extend([total_reward] * len(log_prob)) # we'll assign to all actions the total reward

      optimizer.zero_grad()

      log_probs_batch_tensor = torch.stack(log_probs_batch, dim=0)
      rewards_tensor = torch.tensor(rewards_batch)

      assert len(log_probs_batch_tensor) == len(rewards_tensor)

      if normilize_the_rewards:
        rewards_tensor -= rewards_tensor.mean()
        rewards_tensor /= rewards_tensor.std()

      timesteps += len(log_probs_batch_tensor)

      loss = -log_probs_batch_tensor @ rewards_tensor
      writer.add_scalar("Mean Rewards/train", np.mean(total_rewards), timesteps)
      writer.add_scalar("Loss/train", loss, timesteps)

      loss.backward()

      optimizer.step()

      pbar.update(len(rewards_tensor))

  writer.flush()


def main():

  random.seed(1)
  np.random.seed(1)
  torch.manual_seed(1)

  env = gym.make('CartPole-v1')

  env.reset(seed=1)

  # state and obs/observations are used in this example interchangably.

  state_space = env.observation_space.shape[0]
  action_space = env.action_space.n

  agent = StateToActionLogProb(state_space, action_space)

  mean_reward_before_train = evaluate(env, agent, 100)
  print("before training")
  print(f'{mean_reward_before_train=}')

  train(env, agent, total_timesteps=2_000_000)

  mean_reward_after_train = evaluate(env, agent, 100)
  print("after training")
  print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
  main()
