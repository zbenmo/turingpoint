from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import gymnasium as gym

from turingpoint.gymnasium_utils import (
  EnvironmentParticipant,
  GymnasiumAssembly
)
from turingpoint.sb3_utils import AgentParticipant
from turingpoint.utils import Collector


def main():
  env = gym.make('CartPole-v1')
  agent = PPO(MlpPolicy, env, verbose=0) # use verbose=1 for debugging

  def evaluate(num_episodes: int) -> float:
    rewards_collector = Collector(['reward'])

    assembly = GymnasiumAssembly(env, [
      AgentParticipant(agent, deterministic=True),
      EnvironmentParticipant(env),
      rewards_collector
    ])

    for _ in range(num_episodes):
      _ = assembly.launch()
      # Note we're not clearing the rewards in 'rewards_collector', and so we continue to collect.

    total_reward = sum(x['reward'] for x in rewards_collector.get_entries())

    return total_reward / num_episodes

  def train(total_timesteps):
    agent.learn(total_timesteps=total_timesteps, progress_bar=True) # The agent here learns from its internal Gym environment.
    # We could use a loop with participants here also for training, yet this is not shown here.

  mean_reward_before_train = evaluate(100)
  print("before training")
  print(f'{mean_reward_before_train=}')

  train(total_timesteps=10_000)

  mean_reward_after_train = evaluate(100)
  print("after training")
  print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
  main()