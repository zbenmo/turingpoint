from .definitions import Participant
import stable_baselines3.common


class AgentParticipant(Participant):
  def __init__(self,
               agent: stable_baselines3.common.base_class.BaseAlgorithm,
               **kwargs):
    self.agent = agent
    self.kwargs = kwargs

  def __call__(self, parcel: dict) -> None:
    obs = parcel['obs']
    action, _state = self.agent.predict(obs, **self.kwargs)
    parcel['action'] = action
