import stable_baselines3.common


class AgentParticipant:
  def __init__(self,
               agent: stable_baselines3.common.base_class.BaseAlgorithm,
               **kwargs):
    self._agent = agent
    self._kwargs = kwargs

  def __call__(self, parcel: dict) -> None:
    obs = parcel['obs']
    # action_mask = parcel['action_mask']
    action, _state = self._agent.predict(obs, **self._kwargs)
    parcel['action'] = action
