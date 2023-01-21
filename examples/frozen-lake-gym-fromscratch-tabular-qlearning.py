# based on HuggingFace RL course (unit 2)

import gym
import numpy as np
from tqdm import tqdm

from turingpoint.gym_utils import (
  EnvironmentParticipant,
  GymAssembly
)


# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7          # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"     # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate
eval_seed = []               # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability
decay_rate = 0.0005            # Exponential decay rate for exploration prob


def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state, :])

    return action


def epsilon_greedy_policy(Qtable, state, epsilon):
    explore = np.random.binomial(1, epsilon)
    if explore:
        # Take a random action
        action = np.random.randint(0, Qtable.shape[1])
    else:
        # Take the action with the highest value given a state
        action = np.argmax(Qtable[state, :])

    return action


def main():

  env = gym.make('FrozenLake-v1', desc=None,
                  map_name="4x4", is_slippery=False)

  # state and obs/observations are used in this example interchangably.

  state_space = env.observation_space.n
  action_space = env.action_space.n
  Qtable_frozenlake = initialize_q_table(state_space, action_space)

  def evaluate(num_episodes: int) -> float:
    total_reward = 0

    def agent(parcel: dict) -> None:
      obs = parcel['obs']
      action = greedy_policy(Qtable_frozenlake, obs)
      parcel['action'] = action

    def bookkeeping(parcel: dict) -> None:
      nonlocal total_reward

      reward = parcel['reward']
      total_reward += reward

    assembly = GymAssembly(env, [agent, EnvironmentParticipant(env), bookkeeping])

    for _ in range(num_episodes):
      _ = assembly.launch()

    return total_reward / num_episodes

  def train(num_episodes: n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps):
    epsilon = None
    steps = None

    def agent(parcel: dict) -> None:
      obs = parcel['obs']
      action = epsilon_greedy_policy(Qtable_frozenlake, obs, epsilon)
      parcel['action'] = action

    def learning(parcel: dict) -> None:
      obs = parcel['obs']
      action = parcel['action']
      reward = parcel['reward']
      new_obs = parcel['new_obs']
      target_value = (
        reward + gamma * np.max(Qtable_frozenlake[new_obs, :])
      )
      Qtable_frozenlake[obs][action] = (
        Qtable_frozenlake[obs][action]
        + learning_rate * (target_value - Qtable_frozenlake[obs][action])
      )

    def end_iteration(parcel: dict) -> None:
      nonlocal steps
      
      parcel['obs'] = parcel['new_obs']
      del parcel['new_obs']
      del parcel['reward']
      del parcel['info']
      steps += 1
      parcel['done'] = parcel.get('done', False) or (steps == max_steps)

    assembly = GymAssembly(env, [
      agent,
      EnvironmentParticipant(env, save_obs_as="new_obs"),
      learning,
      end_iteration
    ])

    for episode in tqdm(range(num_episodes), desc="train"):
      epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-decay_rate * episode)
      steps = 0
      _ = assembly.launch()

  train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps)

  mean_reward_after_train = evaluate(100)
  print("after training")
  print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
    main()