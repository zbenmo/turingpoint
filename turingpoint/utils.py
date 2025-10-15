from functools import wraps
from pprint import pprint
import time
from typing import Generator, List, Sequence
from tqdm import tqdm
import numpy as np

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


def call_after_every(parcel: dict, every_x_steps: int, protected: Participant):
   """Use this participant-like to wrap another participant that should be called only every 'every_x_step' steps.
   It relies on 'step' being in the parcel."""
   if (parcel['step'] + 1) % every_x_steps == 0:
      protected(parcel)


def skip_first_n_steps(parcel: dict, n: int, protected: Participant):
    """Use this participant-like to wrap another participant that should be called only after the first n steps.
    It relies on 'step' being in the parcel."""
    if parcel['step'] >= n:
        protected(parcel)


def discounted_reward_to_go(rewards: Sequence[float], gamma=1.0) -> Sequence[float]:
    """Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
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

    def __call__(self, parcel: dict):
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


# Taken from CoPilot

def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards (np.array): Rewards from the environment.
        values (np.array): Value function estimates.
        gamma (float): Discount factor.
        lambda_ (float): GAE coefficient.

    Returns:
        np.array: Computed advantage estimates.
    """
    assert len(values) - len(rewards) == 1, f'{len(values)=}, {len(rewards)=}'

    advantages = np.zeros_like(rewards)
    last_advantage = 0
 
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = delta + gamma * lambda_ * last_advantage
        last_advantage = advantages[t]

    return advantages

# # Example usage:
# rewards = np.array([1, 2, 3, 4, 5])
# values = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5])  # Last value is estimated for bootstrapping
# gamma = 0.99
# lambda_ = 0.95

# # advantages = compute_gae(rewards, values, gamma, lambda_)


def track_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        before = time.time_ns()
        ret = func(*args, **kwargs)
        ns_elapsed = time.time_ns() - before
        wrapper.times_called += 1
        wrapper.ns_elapsed += ns_elapsed
        return ret 
    wrapper.times_called = 0
    wrapper.ns_elapsed = 0
    return wrapper