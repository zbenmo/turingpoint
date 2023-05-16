import gymnasium as gym
import numpy as np
import pettingzoo as pz
from pettingzoo.classic import tictactoe_v3
from turingpoint.definitions import Participant
import turingpoint.sb3_utils as sb3_utils
import turingpoint.pz_utils as pz_utils
import turingpoint.utils as tp_utils
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy, MultiInputPolicy
import logging
# from sb3_contrib import MaskablePPO


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def skip_me_if_I_am_done(participant: Participant):
    def wrapped(parcel: dict):
        terminated = parcel.get('terminated', False)
        truncated = parcel.get('truncated', False)
        if terminated or truncated:
            parcel['action'] = None
        else:
            participant(parcel)
    return wrapped


def pick_a_free_square(parcel: dict):
    action_mask = parcel['obs']['action_mask'] # parcel['action_mask']
    possible_actions = np.where(action_mask == 1)[0]
    parcel['action'] = np.random.choice(possible_actions)


def ensure_a_valid_action(parcel: dict):
    action_mask = parcel['obs']['action_mask'] # parcel['action_mask']
    desired_action = parcel['action'] 
    if action_mask[desired_action]:
        return # we'll stick with the current action
    # otherwise "pick a free square"
    possible_actions = np.where(action_mask == 1)[0]
    parcel['action'] = np.random.choice(possible_actions)


def dummy_gym_env_based_on_pettingzoo(pz_env: pz.AECEnv, agent_id: str) -> gym.Env:
    """This function is needed here as of the interface of sb3, it needs a Gymnasium environment in its constructor,
    to take from it the observation_space and the action_space.
    Note that we only take the observation and drop the action_mask. 
    """
    class Env(gym.Env):
        def __init__(self):
            self.observation_space = pz_env.observation_space(agent_id) # ['observation']
            self.action_space = pz_env.action_space(agent_id)
    return Env()


def extract_obs_and_action_mask(parcel: dict):
  """The PZ tictactoe_v3 environment returns 'action_mask' inside
  the observation, and next to it we have 'observation' which is the state of the board.
  """
  observation, action_mask = (parcel['obs'][x] for x in ['observation', 'action_mask'])
  parcel['obs'] = observation # overriding the previous value
  parcel['action_mask'] = action_mask


def main():
    env = tictactoe_v3.env(render_mode="human")
    env_participant = pz_utils.AECEnvParticipant(env)

    ppo = PPO(
        MultiInputPolicy, # MlpPolicy,
        dummy_gym_env_based_on_pettingzoo(env, "player_1"),
        verbose=0
    ) # use verbose=1 for debugging
    player = sb3_utils.AgentParticipant(ppo)
    other_player = pick_a_free_square

#     our_player_turn = tp_utils.Sequence([
# #        extract_obs_and_action_mask,
#     ])

#     other_player_turn = tp_utils.Sequence([
#     ])

    assembly = pz_utils.AECAssembly(env, [
        skip_me_if_I_am_done(tp_utils.Sequence([player, ensure_a_valid_action])),
        env_participant,
        skip_me_if_I_am_done(other_player),
        env_participant
    ])

    assembly.launch()


if __name__ == "__main__":
    main()