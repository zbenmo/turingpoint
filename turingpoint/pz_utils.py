from typing import Generator, List, Union
from turingpoint.assembly import Assembly
from turingpoint.definitions import Participant
from pettingzoo import AECEnv
from pettingzoo import ParallelEnv
import logging


# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


# class _UpdateParcelFromLast():



class AECEnvParticipant:
  def __init__(
      self,
      env: AECEnv,
      save_obs_as="obs",
      # extract_action_mask_and_observation=True
  ):
    self._env = env
    self._save_obs_as = save_obs_as
    # self._extract_action_mask_and_observation = extract_action_mask_and_observation

  def __call__(self, parcel: dict) -> None:
    action = parcel['action']
    self._env.step(action)
    self._update_parcel_from_last(parcel, self._save_obs_as)

  def _update_parcel_from_last(self, parcel: dict, save_obs_as="obs"):
    parcel.update(
      zip([save_obs_as, 'reward', 'terminated', 'truncated', 'info'], self._env.last())
    )
    # if self._extract_action_mask_and_observation:
    #   obs, action_mask = (parcel[save_obs_as][x] for x in ['observation', 'action_mask'])
    #   parcel[save_obs_as] = obs
    #   parcel['action_mask'] = action_mask


class RenderParticipant:
  def __init__(self, env: Union[AECEnv, ParallelEnv]):
    self._env = env

  def __call__(self, _: dict):
    self._env.render()


class AECAssembly(Assembly):
  def __init__(
      self,
      env: AECEnv,
      participants_list: List[Participant],
      # extract_action_mask_and_observation=True
    ):
    self._env = env
    self._participants_list = participants_list
    # self._extract_action_mask_and_observation = extract_action_mask_and_observation

  def create_initial_parcel(self) -> dict:
    self._env.reset()
    parcel = {
      'agent': self._env.agent_selection
    }
    self._update_parcel_from_last(parcel)
    return parcel

  def _update_parcel_from_last(self, parcel: dict, save_obs_as="obs"):
    parcel.update(
      zip([save_obs_as, 'reward', 'terminated', 'truncated', 'info'], self._env.last())
    )
    # if self._extract_action_mask_and_observation:
    #   obs, action_mask = (parcel[save_obs_as][x] for x in ['observation', 'action_mask'])
    #   parcel[save_obs_as] = obs
    #   parcel['action_mask'] = action_mask

  def participants(self) -> Generator[Participant, None, None]:
    done = False

    def check_done(parcel: dict) -> None:
      "a helper participant for checking for termination condition"

      nonlocal done
      done = parcel['terminated'] or parcel['truncated'] 

    while not done:
      yield from self._participants_list
      yield check_done
