from abc import ABC, abstractmethod
from typing import Callable, TypeVar


Agent = TypeVar("Agent")
Quality = TypeVar("Quality")


class SelfPlay(ABC):
  """
  Template class for training with self play.
  An implementation shall take into account the following considerations:

  * Bootstrap the agent or load an existing model.
  * Bootstrap the opponents (from the agent itself or load existing agents).
  * Maintaining the pool of best, so far, agents. Deciding how many to keep and clearing extra ones.
  * Selection of the opponent among the available in the pool (ex. randomizing or alternative logic).
  * Deciding when to stop a training session againt a specific opponent.
  * Deciding when to stop the whole training session (when to return from the self play launch call).
  * When to save the trained model, making this copy potentially a future opponent
  (while also saving work and making the trained agent available for the actual deployment).
  """

  @abstractmethod
  def fetch_agent_to_train(self) -> Agent:
    pass

  @abstractmethod
  def save_agent(self, agent: Agent):
    pass

  @abstractmethod
  def fetch_opponent(self) -> Agent:
    pass

  @abstractmethod
  def train_against_agent(self, agent_to_train: Agent, opponent_agent: Agent):
    pass

  @abstractmethod
  def evaluate_agent(self, agent: Agent) -> Quality:
    pass

  def launch(self,
             should_stop: Callable[[], bool],
             should_save: Callable[[], bool]) -> Quality:
    agent_to_train = self.fetch_agent_to_train()
    while not should_stop():
      opponent_agent = self.fetch_opponent()
      self.train_against_agent(agent_to_train, opponent_agent)
      if should_save():
        self.save_agent(agent_to_train)
    return self.evaluate_agent(agent_to_train)
