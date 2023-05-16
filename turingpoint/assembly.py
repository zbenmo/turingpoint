from abc import ABC, abstractmethod
from typing import Generator
from .definitions import *


class Assembly(ABC):

  @abstractmethod
  def create_initial_parcel(self) -> dict:
    pass

  @abstractmethod
  def participants(self) -> Generator[Participant, None, None]:
    pass

  def launch(self) -> dict:
    """The 'launch' function is an episode (or main) loop of the evaluation / training / playing / deploying realm.
    For example if you are running multiple episodes you'll probably be calling this function multiple times as needed.
    
    The initial parcel is created with a call to 'self.create_inital_parcel()',
    which in turn shall, for example, set the initial observations.
    Next this function (launch) calls the parcipitants each at its turn.

    The 'self.parcipitants' generator can be based for example on a list,
    where once the end of the list is reached, the generator goes back to the start of the list.
    
    Given that the 'participants' generator is potentially infinite,
    it is the responsibility of (one or more) of the participants to close the generator.
    Closing the generator from a participant can be achieved by raising a flag or by raising an event. See examples.

    Returns:
      The return value is the parcel as is at the end of the loop's execution.
    """
    
    parcel = self.create_initial_parcel()
    for participant in self.participants():
      participant(parcel)
    return parcel
