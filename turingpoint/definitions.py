from typing import Protocol


class Participant(Protocol):
  """
  A participant is, for example, a player or an agent, but can be also the environment itself.
  A participant can be the learning algorithm, a logger, or anything else that makes sense in your realm.
  """
  def __call__(parcel: dict) -> None:
    """
    All the relevant inputs (observation, action, reward, etc.) are passed using the parcel.
    Outputs should also go into the same parcel.
    """
    ...
