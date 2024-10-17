from pprint import pprint
from typing import Any, Generator, List, Sequence

from turingpoint.definitions import Participant


def print_parcel(parcel: dict) -> None:
  """helper "participant", just prints the parcel to the standard output
  """
  pprint(parcel)


class Collector:
  """Simple "participant" that records specific values from the parcel.
  """

  def __init__(self, keys_to_collect = ['obs', 'action', 'new_obs', 'reward', 'done']):
    self._keys_to_collect = keys_to_collect
    self._entries = [] # TODO: potentially replace with dqueue

  def __call__(self, parcel: dict) -> None:
    new_entry = {k: parcel[k] for k in self._keys_to_collect}
    self._entries.append(new_entry)

  def get_entries(self) -> Generator[dict, None, None]:
    yield from self._entries

  def clear_entries(self) -> None:
    self._entries.clear()


def make_sequence(participants: List[Participant]) -> Participant:
  def wrapped(parcel: dict):
    for participant in participants:
      participant(parcel)
  return wrapped


def discounted_reward_to_go(rewards: Sequence[float], gamma=1.0) -> Sequence[float]:
    """
    Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
    in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.

    Copied from homeworks - Berkeley CS 285
    """
    ret = []
    discounted_return = 0.
    for reward in reversed(rewards):
        discounted_return = discounted_return * gamma + reward
        ret.insert(0, discounted_return)
    return ret
