from typing import Generator, List
from turingpoint.assembly import Assembly
from turingpoint.definitions import Participant
import stable_baselines3.common
import gym


class AgentParticipant(Participant):
  def __init__(self, agent: stable_baselines3.common.base_class.BaseAlgorithm):
    self.agent = agent

  def __call__(self, parcel: dict) -> None:
    obs = parcel['obs']
    action, _state = self.agent.predict(obs, deterministic=True)
    parcel['action'] = action


class EnvironmentParticipant(Participant):
  def __init__(self, env: gym.Env):
    self.env = env

  def __call__(self, parcel: dict) -> None:
    action = parcel['action']
    obs, reward, done, info = self.env.step(action)
    # this version of gym is still with done (if you have terminate + truncated) modify it accordingly
    parcel['obs'] = obs
    parcel['reward'] = reward
    parcel['done'] = done
    parcel['info'] = info


class GymAssembly(Assembly):
  def __init__(self, env: gym.Env, participants_list: List[Participant]):
    self.env = env
    self.participants_list = participants_list

  def create_initial_parcel(self) -> dict:
    obs = self.env.reset()
    parcel = {
      'obs': obs
    }
    return parcel

  def participants(self) -> Generator[Participant, None, None]:
    done = False

    def check_done(parcel: dict) -> None:
      "a helper participant for checking for 'done' in the parcel"

      nonlocal done
      done = parcel.get('done', False)

    while not done:
      yield from self.participants_list
      yield check_done
