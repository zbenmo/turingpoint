from pprint import pprint
from typing import Generator, List, Sequence
from tqdm import tqdm

from turingpoint.definitions import Participant, Done


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


class StepsTracker:
    """Participant that shall maintain 'step' in the parcel.
    Use it as a context manager as there is also a progress bar associated with it, which should be closed."""
    def __init__(self, total_timesteps, desc):
        self.total_timesteps = total_timesteps
        self.desc = desc
        self.pbar = None

    def __enter__(self):
        self.pbar = tqdm(total=self.total_timesteps, desc=self.desc)
        return self

    def __call__(self, parcel):
        step = parcel.get('step', None)
        step = 0 if step is None else step + 1
        parcel['step'] = step
        if step >= self.total_timesteps:
            raise Done
        self.pbar.update(1)

    def __exit__(self, exc_type, exc_value, traceback):
        self.pbar.close()


class ReplayBufferCollector:
    """This is a collector based on a list. It is a participant that just copies entries.
    It is meant to be used as a replay buffer with a simple logic that keeps the newest entries.
    """
    def __init__(self, collect, max_entries=10_000):
        self.collect = list(collect)
        self.max_entries = max_entries
        self.replay_buffer = []

    def __call__(self, parcel: dict):
        self.replay_buffer.append({
            k: parcel[k] for k in self.collect
        })
        del self.replay_buffer[0:-self.max_entries]
