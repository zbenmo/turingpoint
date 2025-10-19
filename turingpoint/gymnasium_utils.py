from turingpoint import Done
import gymnasium as gym


def call_reset(parcel, *, env: gym.Env, **kwargs):
  parcel.update(zip(['obs', 'info'], env.reset(**kwargs)))


def call_reset_done(parcel, *, vec_env: gym.vector.VectorEnv, **kwargs):
  # good for SyncVectorEnv. Not sure regarding AsyncVectorEnv
  terminated = parcel['terminated']
  truncated = parcel['truncated']
  obs = parcel['obs']
  info = parcel['info'] # given as a dict of lists (rather than list of dict)
  for i, done in enumerate(terminated | truncated):
    if not done:
      continue
    ob, inf = vec_env.envs[i].reset(**kwargs)
    obs[i] = ob
    for k, val in info.items():
      val[i] = inf.get(k, None)


def call_step(parcel, *, env: gym.Env, save_obs_as="obs"):
  action = parcel['action']
  parcel.update(zip(
    [save_obs_as, 'reward', 'terminated', 'truncated', 'info'],
    env.step(action)
  ))


def check_done(parcel: dict):
  if parcel.get('terminated', False) or parcel.get('truncated', False):
    raise Done


def call_render(parcel, *, env: gym.Env, **kwargs):
  env.render(**kwargs)
