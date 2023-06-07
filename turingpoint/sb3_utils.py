import stable_baselines3.common


def call_predict(parcel, *,
                  agent: stable_baselines3.common.base_class.BaseAlgorithm,
                  **kwargs):
  obs = parcel['obs']
  parcel.update(zip(['action', '_state'], agent.predict(obs, **kwargs)))
