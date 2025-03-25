import functools
import itertools
import random
from typing import Dict

import numpy as np
import torch

import gymnasium as gym
# import spinup
from tqdm import trange

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint.tensorboard_utils as tp_tb_utils
import turingpoint as tp

from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def get_action(parcel: Dict, *, agent):
    obs = parcel['obs']
    action, _states = agent.predict(obs)
    parcel['action'] = action


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



def main():

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    env = gym.make('Humanoid-v4') # gym.make('Humanoid-v5')
 
    env.reset(seed=1)

    # env = gym.make("Pendulum-v1", render_mode="rgb_array")


    # state and obs/observations are used in this example interchangably.

    # state_space = env.observation_space.shape[0]
    # action_space = env.action_space.n

    # agent = spinup.ddpg_pytorch()

    # agent = StateToQValues(state_space, action_space)
    # target_critic = StateToQValues(state_space, action_space)



    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = SAC("MlpPolicy", env, action_noise=action_noise, learning_rate=1e-5) # , verbose=1)
    # vec_env = model.get_env()

    mean_reward_before_train = evaluate(env, model, 100)
    print("before training")
    print(f'{mean_reward_before_train=}')

    model.learn(total_timesteps=100_000, log_interval=10, progress_bar=True)
    # model.save("ddpg_pendulum")

    # del model # remove to demonstrate saving and loading

    # model = DDPG.load("ddpg_pendulum")

    # train(env, agent, target_critic, total_timesteps=1_000_000)

    mean_reward_after_train = evaluate(env, model, 100)
    print("after training")
    print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
    main()
