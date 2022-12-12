from typing import Protocol, Generator, Any
from abc import ABC


class Observation(Protocol):
  ...


class Action(Protocol):
  ...


class Agent(ABC):
  def __init__(self):
    self.agent = self.being()
    next(self.agent)

  def react(self, observation: Observation) -> Action:
    return self.agent.send(observation)

  def being(self) -> Generator[Action, Observation, Any]:
    ...


class MyAgent(Agent):
  def being(self) -> Generator[Action, Observation, Any]:
    obs = yield None
    while True:
      if obs == 'RIP':
        break
      obs = yield 'go_left'
    return "Hope I'll get to heaven."




def main():
  agent = MyAgent()
  position = 5
  try:
    while True:
      print(f'{position=}')
      action = agent.react(position)
      print(f'{action=}')
      if action == 'go_left':
        position = position - 1
      if position < 1:
        print(agent.react('RIP'))
        break
  except StopIteration as si:
    print(si.value)


if __name__ == "__main__":
  main()
