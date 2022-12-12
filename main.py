from typing import Protocol, Generator, Optional
from abc import ABC


class Observation(Protocol):
  ...


class Action(Protocol):
  ...


class Agent(ABC):
  """
  An abstract agent. You need to impelement the '_being' generator.
  In the implementation of '_being' make sure to have the first line "obs = yield None".
  Having the first line in '_being' read "obs = yield None" is needed make sure the first thing that happens is that the agent receives
   an observation.
  The "environment" should use the agent's 'react' function rather than directly calling '_being'.  
  """
  def __init__(self):
    self.reset()

  def react(self, observation: Observation) -> Optional[Action]:
    """
    Call this function from your main loop / environment. The return value is None when the agent is not any more active.
    """
    try:
      return self._mind.send(observation)
    except StopIteration:
      return None

  def _being(self) -> Generator[Action, Observation, None]:
    ...

  def reset(self):
    self._mind = self._being()
    next(self._mind)


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

  def reset(self):
    super().reset()
    if hasattr(self, 'final_wish'):
      del self.final_wish


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


def two_episodes_example():
  agent = MyAgent()
  for episode in range(2):
    print(f'{episode=}')
    print("--")
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
    print()
    agent.reset()


class MyRLAgent(Agent):
  def __init__(self, default_action: str = "go_left"):
    super().__init__()
    self.default_action = default_action
    self.total_reward = 0

  def _being(self) -> Generator[Action, Observation, None]:
    obs = yield None
    while True:
      if obs == 'done':
        break
      if isinstance(obs, dict):
        self.total_reward += obs['reward']
      obs = yield self.default_action
    self.final_wish = "To be recognized for the total reward that I've collected."

  def reset(self):
    super().reset()
    self.total_reward = 0
    if hasattr(self, 'final_wish'):
      del self.final_wish


def reward_done_example():
  agent = MyRLAgent()
  position = 5
  print(f'{position=}')
  action = agent.react(position)
  while True:
    print(f'{action=}')
    if action == 'go_left':
      position = position - 1
      print(f'{position=}')
    if position < 1:
      agent.react('done')
      break
    else:
      action = agent.react({
        'reward': +1,
        'position': position
      })
  print(f'The agent collected {agent.total_reward} total reward.')


def main():
  print("one_agent_example")
  print("-----------------")
  one_agent_example()
  print()
  print("two_agents_example")
  print("-----------------")
  two_agents_example()
  print()
  print("two_episodes_example")
  print("--------------------")
  two_episodes_example()
  print()
  print("reward_done_example")
  print("--------------------")
  reward_done_example()
  print()


if __name__ == "__main__":
  main()
