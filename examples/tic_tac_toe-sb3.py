import functools
import itertools
import gymnasium as gym
import numpy as np
import pettingzoo as pz
from pettingzoo.classic import tictactoe_v3
from turingpoint.definitions import Participant
import turingpoint.sb3_utils as sb3_utils
import turingpoint.pz_utils as pz_utils
import turingpoint.utils as tp_utils
import turingpoint as tp
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy, MultiInputPolicy
# import logging
# from sb3_contrib import MaskablePPO


# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def skip_me_if_I_am_done(participant: Participant) -> Participant:
    def wrapped(parcel: dict):
        terminated = parcel.get('terminated', False)
        truncated = parcel.get('truncated', False)
        if terminated or truncated:
            parcel['action'] = None
        else:
            participant(parcel)
    return wrapped


@skip_me_if_I_am_done
def pick_a_free_square(parcel: dict):
    action_mask = parcel['obs']['action_mask']
    possible_actions = np.where(action_mask == 1)[0]
    parcel['action'] = np.random.choice(possible_actions)


def ensure_a_valid_action(parcel: dict):
    action_mask = parcel['obs']['action_mask']
    desired_action = parcel['action'] 
    if action_mask[desired_action]:
        return # we'll stick with the current action
    # otherwise "pick a free square"
    possible_actions = np.where(action_mask == 1)[0]
    parcel['action'] = np.random.choice(possible_actions)


def dummy_gym_env_based_on_pettingzoo(pz_env: pz.AECEnv, agent_id: str) -> gym.Env:
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
    env = tictactoe_v3.env(render_mode="human")
    env_participant = functools.partial(pz_utils.call_step, env=env)

    ppo = PPO(
        MultiInputPolicy, # MlpPolicy,
        dummy_gym_env_based_on_pettingzoo(env, "player_1"),
        verbose=0
    ) # use verbose=1 for debugging
    player = functools.partial(sb3_utils.call_predict, agent=ppo)
    player = tp_utils.make_sequence([player, ensure_a_valid_action])
    player = skip_me_if_I_am_done(player)
    other_player = pick_a_free_square # it is already wrapped in 'skip_me_if_I_am_done'

    def assert_player_1_turn(parcel: dict):
        assert parcel['agent'] == 'player_1'

    def assert_player_2_turn(parcel: dict):
        assert parcel['agent'] == 'player_2'

    def get_participants():
        yield functools.partial(pz_utils.call_reset, env=env)
        yield from itertools.cycle([
            assert_player_1_turn,
            player,
            env_participant,
            assert_player_2_turn,
            other_player,
            env_participant,
            pz_utils.check_done
        ])

    assembly = tp.Assembly(get_participants)
    
    assembly.launch()


if __name__ == "__main__":
    main()