from pprint import pprint
from typing import Any, Generator, List

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
