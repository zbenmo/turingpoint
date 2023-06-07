from typing import Callable, Iterable, Iterator, Union
from .definitions import Participant, Done


class Assembly:
  """
  This class contains a loop that goes over the participants and passes the parcel among those.
  """

  def __init__(self,
               get_patricipants: Callable[[], Union[Iterator[Participant], Iterable[Participant]]]):
    """ constructor

    Args:
      get_participants: A callable that returns an iterable or an iterator for the participants.
    """

    self._get_participants = get_patricipants

  def launch(self):
    """The 'launch' function is an episode (or main) loop
    of the evaluation / training / playing / deploying realm.
    For example if you are running multiple episodes you'll probably be
    calling this function multiple times as needed.
    
    The initial empty parcel is created here.
    Next this function (launch) calls the parcipitants each at its turn.

    Given that the 'get_patricipants' may potentially return iterators that are infinite,
    it is the responsibility of (one or more) of the participants to stop the iterator.
    One option to stop the iterator is to raise turingpoint.Done exception which is handled silently here.

    Returns:
      The return value is the parcel as is at the end of the loop's execution.
    """

    parcel = {}
    try:
      for participant in self._get_participants():
        if participant is None:
          continue
        participant(parcel)
    except Done:
      pass
    return parcel
