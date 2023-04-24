from typing import Generator, List
from turingpoint import Assembly, Participant
import gymnasium as gym


class EnvironmentParticipant(Participant):
  def __init__(self, env: gym.Env, save_obs_as="obs"):
    self._env = env
    self._save_obs_as = save_obs_as

  def __call__(self, parcel: dict) -> None:
    action = parcel['action']
    obs, reward, terminated, truncated, info = self._env.step(action)
    parcel[self._save_obs_as] = obs
    parcel['reward'] = reward
    parcel['terminated'] = terminated
    parcel['truncated'] = truncated
    parcel['info'] = info


class RenderParticipant(Participant):
  def __init__(self, env: gym.Env):
    self._env = env

  def __call__(self, _: dict):
    self._env.render()


class GymnasiumAssembly(Assembly):
  def __init__(self, env: gym.Env, participants_list: List[Participant]):
    self._env = env
    self._participants_list = participants_list

  def create_initial_parcel(self) -> dict:
    obs, info = self._env.reset()
    parcel = {
      'obs': obs,
      'info': info
    }
    return parcel

  def participants(self) -> Generator[Participant, None, None]:
    done = False

    def check_done(parcel: dict) -> None:
      "a helper participant for checking for 'done' in the parcel"

      nonlocal done
      done = parcel.get('terminated', False) or parcel.get('truncated', False)

    while not done:
      yield from self._participants_list
      yield check_done
