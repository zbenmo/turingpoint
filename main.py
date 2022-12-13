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
    next(self._mind) # needed to kick-off the generator


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


class PreditorAgent(MyAgent):
  def __init__(self, default_action: str = "go_left"):
    super().__init__(default_action)

  def _being(self) -> Generator[Action, Observation, None]:
    obs = yield None
    while True:
      if obs == 'RIP':
        break
      action = "go_right" if obs.get('pray', None) is not None else self.default_action 
      obs = yield action
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
  agent1 = MyAgent()
  agent2 = PreditorAgent()
  position1 = 5
  position2 = 1
  while True:
    if agent1:
      print(f'{position1=}')
    print(f'{position2=}')
    if agent1:
      action1 = agent1.react(position1)
      print(f'{action1=}')
    obs_agent2 = {
      'you': position2
    }
    if agent1 is not None:
      obs_agent2.update({
        'pray': position1
      })
    action2 = agent2.react(obs_agent2)
    print(f'{action2=}')
    if action1 == 'go_left':
      position1 += -1
    if action2 == 'go_right':
      position2 += +1
    elif action2 == 'go_left':
      position2 += -1
    if agent1:
      if position1 <= position2:
        agent1.react('RIP')
        agent1 = None
    if position2 < 0:
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
    self.episode = []

  def _being(self) -> Generator[Action, Observation, None]:
    obs = yield None
    while True:
      self.episode.append(obs)
      self.total_reward += obs.get('reward', 0)
      if obs.get('done', False):
        break
      action = self.default_action
      self.episode.append(action)
      obs = yield action
    self.final_wish = "To be recognized for the total reward that I've collected."

  def reset(self):
    super().reset()
    self.total_reward = 0
    self.episode = []
    if hasattr(self, 'final_wish'):
      del self.final_wish


def reward_done_example():
  agent = MyRLAgent()
  position = 5
  print(f'{position=}')
  action = agent.react({
    'position': position
  })
  while True:
    print(f'{action=}')
    if action == 'go_left':
      position = position - 1
      print(f'{position=}')
    if position < 1:
      agent.react({
        'done': True,
        'reward': -10
      })
      break
    else:
      action = agent.react({
        'reward': +1,
        'position': position
      })
  print(f'The agent collected {agent.total_reward} total reward.')
  print(f"Judging from the agent's recollection, the episode is {agent.episode}")


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
