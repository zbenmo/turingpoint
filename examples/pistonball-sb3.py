import functools
import itertools
import gymnasium as gym
import numpy as np
import pandas as pd
import pettingzoo as pz
from pettingzoo.butterfly import pistonball_v6
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy, MultiInputPolicy

from turingpoint import pz_utils, sb3_utils
import turingpoint as tp
from turingpoint.utils import Collector


def dummy_gym_env_based_on_pettingzoo(pz_env: pz.ParallelEnv, agent_id: str) -> gym.Env:
    """This function is needed here as of the interface of sb3;
    it needs a Gymnasium environment in its constructor,
    to take from it the observation_space and the action_space.
    """
    class Env(gym.Env):
        def __init__(self):
            self.observation_space = pz_env.observation_space(agent_id)
            self.action_space = pz_env.action_space(agent_id)
    return Env()


def main():
    env = pistonball_v6.parallel_env(render_mode="human")
    env_participant = functools.partial(
        pz_utils.call_step_parallel,
        env=env,
        save_observations_as="new_observations"
    )

    pistons_agent = {}
    pistons = {}
    for id in range(20):
        piston_id = f'piston_{id}'
        pistons_agent[piston_id] = PPO(
            MlpPolicy, #MultiInputPolicy, # MlpPolicy,
            dummy_gym_env_based_on_pettingzoo(env, piston_id),
            verbose=0
        ) # use verbose=1 for debugging
        pistons[piston_id] = functools.partial(sb3_utils.call_predict,
                                               agent=pistons_agent[piston_id])

    def all_pistons(parcel: dict):
        nonlocal pistons

        parcel['actions'] = {}
        for piston_id in parcel['agents']:
            parcel['obs'] = parcel['observations'][piston_id]
            pistons[piston_id](parcel)
            parcel['actions'][piston_id] = parcel.pop('action')
        del parcel['obs']

    collector = Collector([
        'observations',
        'actions',
        'rewards',
        'new_observations',
        'terminations',
        'truncations',
        'infos',
    ])

    def update_observations(parcel: dict):
        parcel['observations'] = parcel.pop('new_observations')

    def get_participants():
        yield functools.partial(pz_utils.call_reset_parallel, env=env)
        yield from itertools.cycle([
            all_pistons,
            env_participant,
            collector,
            pz_utils.check_done_parallel,
            update_observations
        ])

    assembly = tp.Assembly(get_participants)
    
    assembly.launch()

    piston_id = 'piston_0'

    ppo = pistons_agent[piston_id]

    ppo = PPO(
            MlpPolicy, #MultiInputPolicy, # MlpPolicy,
            dummy_gym_env_based_on_pettingzoo(env, piston_id),
            verbose=0
        ) # use verbose=1 for debugging

    df = pd.DataFrame.from_records(collector.get_entries())

    ppo.rollout_buffer.add(
        obs=df.observations.apply(lambda obs: obs[piston_id]).values,
        action=df.actions.apply(lambda actions: actions[piston_id]).values,
        reward=df.rewards.apply(lambda rewards: rewards[piston_id]).values,
        episode_start=None,
        value=None,
        log_prob=np.array(0),
        # reward=df.actions.apply(lambda rewards: rewards[piston_id]),
    )
    ppo.train()


if __name__ == "__main__":
    main()