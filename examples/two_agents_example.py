import itertools
from typing import Generator, Optional
from dataclasses import dataclass
from turingpoint import Assembly, Participant, Done
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

  def agent1(parcel: dict) -> None:
    """
    participant "prey"
    """
    parcel['action_agent1'] = "go_left"

  def agent2(parcel: dict) -> None:
    """
    participant "hunter"
    """
    action = "go_right" if parcel['obs'].agent1 else "go_left"
    parcel['action_agent2'] = action 

  def environment(parcel: dict) -> None:
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
    if state.position_agent2 < 0:
      parcel['done'] = True
      # Here we'll only report on the end of the whole episode for agent2,
      # but we could've report about agent1 also and more.
    set_observation(parcel)

  def check_done(parcel: dict) -> None:
    "a helper participant for checking for 'done' in the parcel"

    # below is just to clean a bit for better output and for making sure no action is retaken accidentally
    parcel.pop('action_agent1', None)
    parcel.pop('action_agent2', None)

    if parcel.get('done', False):
      raise Done()

  def get_participants():
    yield set_observation
    while True: yield from [
      print_parcel,
      agent1 if state.agent1 else None, # passing None as a participants results in skipping that participant
      agent2,
      environment,
      print_parcel,
      check_done
    ]

  assembly = Assembly(get_participants)

  assembly.launch()


if __name__ == "__main__":
  two_agents_example()
