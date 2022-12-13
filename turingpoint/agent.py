from typing import Generator, Optional
from abc import ABC

from .definitions import *


class Agent(ABC):
  """
  An abstract agent. You need to impelement the '_being' generator.
  In the implementation of '_being' make sure to have the first line "obs = yield None".
  Having the first line in '_being' read "obs = yield None" is needed make sure the first thing that happens is that the agent receives
   an observation.
  The "environment" should use the agent's 'react' function rather than directly calling '_being'.  
  """
  def __init__(self):
    self.reset()

  def react(self, observation: Observation) -> Optional[Action]:
    """
    Call this function from your main loop / environment. The return value is None when the agent is not any more active.
    """
    try:
      return self._mind.send(observation)
    except StopIteration:
      return None

  def _being(self) -> Generator[Action, Observation, None]:
    ...

  def reset(self):
    self._mind = self._being()
    next(self._mind) # needed to kick-off the generator
