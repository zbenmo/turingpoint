import numpy as np
from typing import Generator, Dict, Protocol
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import gym

import turingpoint as tp


class CartPoleAgentObservation(Protocol):
  ...


class CartPoleAction(Protocol):
  ...


class CartPoleAgent(Protocol):
  def observe(obs: CartPoleAgentObservation) -> CartPoleAction:
    ...


class CartPoleEnvironment(Protocol):
  def evaluate_agent(self, num_episodes=100):
    ...


class MyCartPoleAgent(tp.Agent):
    def __init__(self):
        super(MyCartPoleAgent, self).__init__()
        env = gym.make('CartPole-v1')
        self.model = PPO("MlpPolicy", env, verbose=0) # use verbose=1 for debugging
        self.memory = {
            ...
        }

    def _being(self) -> Generator[str, dict, None]:
        obs: CartPoleAgentObservation = yield None
        while True:
            if obs.get('done', False):
                break
            action, _state = self.model.predict(obs.get('gym_obs'), deterministic=True) 
            obs = yield action

    def reset(self):
        super().reset()
        self.memory = {
            ...
        }

    def learn(self, total_timesteps=10_000):
      """
      This method is not part of the "generic" abstract class of tp.Agent.
      """
      self.model.learn(total_timesteps=total_timesteps, progress_bar=True)


class MyCartPoleEnvironment:
  def __init__(self, agent: CartPoleAgent):
    self.agent = agent
    self.env = gym.make('CartPole-v1')

  @classmethod
  def _gym_obs_to_my_obs(cls, gym_obs, reward = None, done = None) -> CartPoleAgentObservation:
    ret = {
      'gym_obs': gym_obs
    }
    if done:
      ret['done'] = True
    if reward:
      ret['reward'] = reward
    return ret

  def evaluate_agent(self, num_episodes=100):
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        self.agent.reset()
        gym_obs = self.env.reset()
        obs = MyCartPoleEnvironment._gym_obs_to_my_obs(gym_obs)
        while not done:
            action = self.agent.observe(obs)
            gym_obs, reward, done, info = self.env.step(action)
            episode_rewards.append(reward)
            # We don't bother here to pass the reward.
            obs = MyCartPoleEnvironment._gym_obs_to_my_obs(gym_obs, done=done)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


def main():
  agent = MyCartPoleAgent()
  environment = MyCartPoleEnvironment(agent)

  print("before training")
  mean_reward_before_train = environment.evaluate_agent(num_episodes=100)

  agent.learn(total_timesteps=10000) # The agent here learns from its internal Gym environment.

  print("after training")
  mean_reward_after_train = environment.evaluate_agent(num_episodes=100)


if __name__ == "__main__":
    main()