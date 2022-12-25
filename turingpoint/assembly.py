from typing import Protocol
from .definitions import *


class GetState(Protocol):
  def __call__() -> State:
    ...


class ScatterObservations(Protocol):
  def __call__(state: State) -> Action:
    ...


class Result(Protocol):
  ...


class ApplyActions(Protocol):
  def __call__(actions: Action) -> Result:
    ...


class ObserveResults(Protocol):
  def __call__(results: Result) -> bool:
    ...


class Assembly:
  def __init__(
    self,
    get_state: GetState,
    scatter_observations: ScatterObservations,
    apply_actions: ApplyActions,
    observe_results: ObserveResults):

    self.get_state = get_state
    self.scatter_observations = scatter_observations
    self.apply_actions = apply_actions
    self.observe_results = observe_results

  def launch(self):
    done = False
    while not done:
      state = self.get_state()
      actions = self.scatter_observations(state)
      results = self.apply_actions(actions)
      done = self.observe_results(results)
