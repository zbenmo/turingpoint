from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import gym

# import turingpoint as tp
from turingpoint.environment import Environment


def main():

  env = gym.make('CartPole-v1')
  agent = PPO(MlpPolicy, env, verbose=0) # use verbose=1 for debugging

  def evaluate(num_episodes: int) -> float:
    total_reward = 0

    state = None

    def get_state():
      return state

    def scatter_observations(state):
      action, _state = agent.predict(state, deterministic=True)
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

  def train(total_timesteps):
    agent.learn(total_timesteps=total_timesteps, progress_bar=True) # The agent here learns from its internal Gym environment.
    # We could use an environment for training, yet this is not shown here.

  mean_reward_before_train = evaluate(100)
  print("before training")
  print(f'{mean_reward_before_train=}')

  train(total_timesteps=10_000)

  mean_reward_after_train = evaluate(100)
  print("after training")
  print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
    main()