from turingpoint.assembly import Assembly
from turingpoint.definitions import Participant
from turingpoint.utils import print_parcel
from typing import List, Generator


def two_episodes_example():
  """
  position is the "state" in this example. There is one agent that keeps going left.
  """
  position: int = 5

  def set_observation(parcel: dict):
    """
    helper to avoid repetition
    """
    parcel['obs'] = position

  class MyAssembly(Assembly):
    def __init__(self, participants_list: List[Participant]):
      self.participants_list = participants_list

    def create_initial_parcel(self) -> dict:
      """
      The initial parcel, the first observation from the environment.
      """
      parcel = {}
      set_observation(parcel)
      return parcel

    def participants(self) -> Generator[Participant, None, None]:
      done = False

      def check_done(parcel: dict) -> None:
        "a helper participant for checking for 'done' in the parcel"

        nonlocal done
        done = parcel.get('done', False)

      while not done:
        yield from self.participants_list
        yield check_done

  def agent(parcel: dict) -> None:
    """
    participant - the agent
    """
    obs = parcel['obs']
    assert obs > 0
    action = 'go_left'
    parcel['action'] = action

  def environment(parcel: dict) -> None:
    """
    participant - the environment
    """
    nonlocal position

    action = parcel['action']
    if action == 'go_left':
      position = position - 1
      set_observation(parcel)
    if position < 1:
      parcel['done'] = True

  assembly = MyAssembly([print_parcel, agent, environment, print_parcel])

  for episode in range(2):
    print(f'{episode=}')
    print("--")
    assembly.launch()
    print()
    position = 5

if __name__ == "__main__":
  two_episodes_example()
