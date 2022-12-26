from typing import List, Generator
from .definitions import Participant


def print_parcel(parcel: dict) -> None:
  """
  helper "participant", just prints the parcel to the standard output
  """
  print(parcel)


def generator_from_list(participants: List[Participant]) -> Generator[Participant, None, None]:
  while True:
    yield from participants


# copied from https://stackoverflow.com/a/2022629/1614089
class Event(list):
    """Event subscription.

    A list of callable objects. Calling an instance of this will cause a
    call to each item in the list in ascending order by index.

    Example Usage:
    >>> def f(x):
    ...     print 'f(%s)' % x
    >>> def g(x):
    ...     print 'g(%s)' % x
    >>> e = Event()
    >>> e()
    >>> e.append(f)
    >>> e(123)
    f(123)
    >>> e.remove(f)
    >>> e()
    >>> e += (f, g)
    >>> e(10)
    f(10)
    g(10)
    >>> del e[0]
    >>> e(2)
    g(2)

    """
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)