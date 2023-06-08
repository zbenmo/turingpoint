from typing import Union
from turingpoint.assembly import Done
from pettingzoo import AECEnv
from pettingzoo import ParallelEnv


def call_reset(parcel: dict, *, env: AECEnv, **kwargs):
  env.reset(**kwargs)
  parcel.update(
    zip(['obs', 'reward', 'terminated', 'truncated', 'info'], env.last())
  )
  parcel['agent'] = env.agent_selection


def call_step(parcel: dict, *, env: AECEnv, save_obs_as="obs"): 
  action = parcel['action']
  env.step(action)
  parcel.update(
    zip([save_obs_as, 'reward', 'terminated', 'truncated', 'info'], env.last())
  )
  parcel['agent'] = env.agent_selection


def call_render(parcel: dict, *, env: Union[AECEnv, ParallelEnv], **kwargs):
  env.render(**kwargs)


def check_done(parcel: dict):
  if parcel.get('terminated', False) or parcel.get('truncated', False):
    raise Done


def call_reset_parallel(parcel: dict, *, env: ParallelEnv, **kwargs):
  parcel["observations"] = env.reset(**kwargs)
  parcel["agents"] = env.agents


def call_step_parallel(parcel: dict, *, env: ParallelEnv, save_observations_as="observations"): 
  actions = parcel['actions']
  parcel.update(
    zip([
      save_observations_as,
      'rewards',
      'terminations',
      'truncations',
      'infos',
    ], env.step(actions))
  )
  parcel["agents"] = env.agents


def check_done_parallel(parcel: dict):
  if len(parcel["agents"]) < 1:
    raise Done
