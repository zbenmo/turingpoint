from pprint import pprint


def print_parcel(parcel: dict) -> None:
  """
  helper "participant", just prints the parcel to the standard output
  """
  pprint(parcel)


class Collector:
  """
  Simple "participant" that records specific values from the parcel into a list (a buffer).
  It is the responsibility of the user to access to the buffer (the entries), to clean the buffer (clean it or set it to a new empty list).
  It is also the responsitibity of the user to format into the desired format as appropriate.
  """

  def __init__(self, keys_to_collect = ['obs', 'action', 'new_obs', 'reward', 'done']):
    self.keys_to_collect = keys_to_collect
    self.entries = [] # TODO: potentially replace with dqueue

  def __call__(self, parcel: dict) -> None:
    new_entry = {k: parcel[k] for k in self.keys_to_collect}
    self.entries.append(new_entry)