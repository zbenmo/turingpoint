from turingpoint import Done
import gymnasium as gym


def call_reset(parcel, *, env: gym.Env, **kwargs) -> dict:
  parcel.update(zip(['obs', 'info'], env.reset(**kwargs)))


def call_step(parcel, *, env: gym.Env, save_obs_as="obs") -> dict:
  action = parcel['action']
  parcel.update(zip(
    [save_obs_as, 'reward', 'terminated', 'truncated', 'info'],
    env.step(action)
  ))


def check_done(parcel: dict):
  if parcel.get('terminated', False) or parcel.get('truncated', False):
    raise Done


def call_render(parcel, *, env: gym.Env, **kwargs) -> dict:
  env.render(**kwargs)
