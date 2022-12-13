from typing import Generator, Dict, Protocol
import gym
import numpy as np
from tqdm import tqdm

# import turingpoint as tp
from turingpoint.environment import Environment


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
    # Randomly generate a number between 0 and 1
    random_num = np.random.random()
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = np.argmax(Qtable[state, :])
    # else --> exploration
    else:
        action = np.random.randint(0, Qtable.shape[1])  # Take a random action

    return action


def main():

  env = gym.make('FrozenLake-v1', desc=None,
                  map_name="4x4", is_slippery=False)

  state_space = env.observation_space.n
  action_space = env.action_space.n
  Qtable_frozenlake = initialize_q_table(state_space, action_space)

  def evaluate(num_episodes: int) -> float:
    total_reward = 0

    state = None

    def get_state():
      return state

    def scatter_observations(state):
      action = greedy_policy(Qtable_frozenlake, state)
      return action

    def apply_actions(action):
      return env.step(action)

    def observe_results(results) -> bool:
      nonlocal state
      nonlocal total_reward

      state, reward, done, info = results
      total_reward += reward
      return done      

    eval_environment = Environment(
      get_state,
      scatter_observations,
      apply_actions,
      observe_results
    )

    for _ in range(num_episodes):
      state = env.reset()
      eval_environment.launch()

    return total_reward / num_episodes

  def train(num_episodes: n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps):
    state = None
    last_action = None

    epsilon = None
    steps = None

    def get_state():
      return state

    def scatter_observations(state):
      nonlocal epsilon

      action = epsilon_greedy_policy(Qtable_frozenlake, state, epsilon)
      return action

    def apply_actions(action):
      nonlocal last_action

      last_action = action
      return env.step(action)

    def observe_results(results) -> bool:
      nonlocal state
      nonlocal steps

      new_state, reward, done, info = results
      target_value = (
        reward + gamma * np.max(Qtable_frozenlake[new_state, :])
      )
      Qtable_frozenlake[state][last_action] = (
        Qtable_frozenlake[state][last_action]
        + learning_rate * (target_value - Qtable_frozenlake[state][last_action])
      )
      state = new_state
      steps += 1
      return done or (steps == max_steps)

    train_environment = Environment(
      get_state,
      scatter_observations,
      apply_actions,
      observe_results
    )

    for episode in tqdm(range(num_episodes), desc="train"):
      epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-decay_rate * episode)
      state = env.reset()
      steps = 0
      train_environment.launch()

  train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps)

  mean_reward_after_train = evaluate(100)
  print("after training")
  print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
    main()