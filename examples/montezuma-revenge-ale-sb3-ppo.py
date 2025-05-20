import functools
import itertools
import random
import ale_py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import CnnPolicy
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import torch
from tqdm import trange

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.sb3_utils as tp_sb3_utils
import turingpoint.utils as tp_utils
import turingpoint as tp


gym_environment = "ALE/MontezumaRevenge-v5"


def make_env(**kwargs) -> gym.Env:
    env = gym.make(gym_environment, frameskip=1, **kwargs) # (210, 160, 3)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=3, # 3 + 1 = 4 ?
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)

    return env


def evaluate(env, agent, num_episodes: int) -> float:

  rewards_collector = tp_utils.Collector(['reward'])

  def get_participants():
    yield functools.partial(tp_gym_utils.call_reset, env=env)
    yield from itertools.cycle([
        functools.partial(tp_sb3_utils.call_predict, agent=agent, deterministic=True),
        functools.partial(tp_gym_utils.call_step, env=env),
        rewards_collector,
        tp_gym_utils.check_done
    ]) 

  evaluate_assembly = tp.Assembly(get_participants)

  for _ in trange(num_episodes, desc='evaluate'):
    _ = evaluate_assembly.launch()
    # Note that we don't clear the rewards in 'rewards_collector', and so we continue to collect.

  total_reward = sum(x['reward'] for x in rewards_collector.get_entries())

  return total_reward / num_episodes


def train(agent, total_timesteps):
  agent.learn(total_timesteps=total_timesteps, progress_bar=True)
  # The agent here learns from its internal Gym environment.
  # We could use a loop with participants also for training, yet this is not shown here.


def main():

  random.seed(1)
  np.random.seed(1)
  torch.manual_seed(1)

  gym.register_envs(ale_py)

  env = make_env()

  env.reset(seed=1)

  agent = PPO(CnnPolicy, env, verbose=0) # use verbose=1 for debugging

  episodes_for_evaluation = 10

  mean_reward_before_train = evaluate(env, agent, episodes_for_evaluation)
  print("before training")
  print(f'{mean_reward_before_train=}')

  train(agent, total_timesteps=1_000)

  mean_reward_after_train = evaluate(env, agent, episodes_for_evaluation)
  print("after training")
  print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
  main()
