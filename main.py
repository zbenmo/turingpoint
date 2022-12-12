from typing import Protocol, Generator, Any
from abc import ABC


class Observation(Protocol):
  ...


class Action(Protocol):
  ...


class Agent(ABC):
  """
  An abstract agent. You need to impelement the 'being' generator.
  In the implementation of '_being' make sure to have the first line "obs = yield None".
  Having the first line in '_being' read "obs = yield None" is needed make sure the first thing that happens is that the agent receives
   an observation.
  The "environment" shall use the agent's 'react' function rather than directly calling '_being'.  
  """
  def __init__(self):
    self.agent = self._being()
    next(self.agent)

  def react(self, observation: Observation) -> Action:
    try:
      return self.agent.send(observation)
    except StopIteration:
      return None

  def _being(self) -> Generator[Action, Observation, None]:
    ...


class MyAgent(Agent):
  def __init__(self, default_action: str = "go_left"):
    super().__init__()
    self.default_action = default_action

  def _being(self) -> Generator[Action, Observation, None]:
    obs = yield None
    while True:
      if obs == 'RIP':
        break
      obs = yield self.default_action
    self.final_wish = "Hope I'll get to heaven."


def one_agent_example():
  agent = MyAgent()
  position = 5
  while True:
    print(f'{position=}')
    action = agent.react(position)
    print(f'{action=}')
    if action == 'go_left':
      position = position - 1
    if position < 1:
      agent.react('RIP')
      break


def two_agents_example():
  agent1 = MyAgent("go_left")
  agent2 = MyAgent("go_right")
  position1 = 5
  position2 = 1
  while True:
    print(f'{position1=}')
    print(f'{position2=}')
    action1 = agent1.react(position1)
    print(f'{action1=}')
    action2 = agent2.react(position2)
    print(f'{action2=}')
    if action1 == 'go_left':
      position1 = position1 - 1
    if action2 == 'go_right':
      position2 = position2 + 1
    if position1 <= position2:
      agent1.react('RIP')
      agent2.react('RIP')
      break


def main():
  print("one_agent_example")
  print("-----------------")
  one_agent_example()
  print()
  print("two_agents_example")
  print("-----------------")
  two_agents_example()
  print()


if __name__ == "__main__":
  main()
