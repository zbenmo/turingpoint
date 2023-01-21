from typing import Generator, Optional
from dataclasses import dataclass
from turingpoint import Assembly, Participant
from turingpoint.utils import print_parcel


@dataclass
class WorldState:
  agent1: bool
  position_agent1: Optional[int]
  position_agent2: int


def two_agents_example():

  state = WorldState(True, 5, 1)

  def set_observation(parcel: dict):
    """
    helper to avoid repetition
    """
    parcel['obs'] = state # potentially use dataclasses.replace(state) to create a copy

  class MyAssembly(Assembly):
    def __init__(self):
      self.participants_list = [print_parcel, self.agent1, self.agent2, self.environment, print_parcel]

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
        # below is just to clean a bit for better output and for making sure no action is retaken accidentally
        parcel.pop('action_agent1', None)
        parcel.pop('action_agent2', None)

      while not done:
        yield from self.participants_list
        yield check_done

    def agent1(self, parcel: dict) -> None:
      """
      participant "prey"
      """
      parcel['action_agent1'] = "go_left"

    def agent2(self, parcel: dict) -> None:
      """
      participant "hunter"
      """
      action = "go_right" if parcel['obs'].agent1 else "go_left"
      parcel['action_agent2'] = action 

    def environment(self, parcel: dict) -> None:
      nonlocal state
      
      if state.agent1:
        if parcel['action_agent1'] == 'go_left':
          state.position_agent1 += -1
      if parcel['action_agent2'] == 'go_right':
        state.position_agent2 += +1
      elif parcel['action_agent2'] == 'go_left':
        state.position_agent2 += -1
      if state.agent1:
        if state.position_agent1 <= state.position_agent2:
          state.agent1 = False
          state.position_agent1 = None
          self.participants_list.remove(self.agent1)
      if state.position_agent2 < 0:
        parcel['done'] = True
        # Here we'll only report on the end of the whole episode for agent2, but we could've report about agent1 also and more.
      set_observation(parcel)

  assembly = MyAssembly()

  assembly.launch()


if __name__ == "__main__":
  two_agents_example()
