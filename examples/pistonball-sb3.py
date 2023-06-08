import functools
import itertools
import gymnasium as gym
import pettingzoo as pz
from pettingzoo.butterfly import pistonball_v6
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy, MultiInputPolicy

from turingpoint import pz_utils, sb3_utils
import turingpoint as tp


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
    env_participant = functools.partial(pz_utils.call_step_parallel, env=env)

    pistons = {}
    for id in range(20):
        piston_id = f'piston_{id}'
        ppo = PPO(
            MlpPolicy, #MultiInputPolicy, # MlpPolicy,
            dummy_gym_env_based_on_pettingzoo(env, piston_id),
            verbose=0
        ) # use verbose=1 for debugging
        pistons[piston_id] = functools.partial(sb3_utils.call_predict, agent=ppo)

    def all_pistons(parcel: dict):
        nonlocal pistons

        parcel['actions'] = {}
        for piston_id in parcel['agents']:
            parcel['obs'] = parcel['observations'][piston_id]
            pistons[piston_id](parcel)
            parcel['actions'][piston_id] = parcel['action']

    def get_participants():
        yield functools.partial(pz_utils.call_reset_parallel, env=env)
        yield from itertools.cycle([
            all_pistons,
            env_participant,
            pz_utils.check_done_parallel
        ])

    assembly = tp.Assembly(get_participants)
    
    assembly.launch()


if __name__ == "__main__":
    main()