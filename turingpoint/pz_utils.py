from turingpoint.assembly import Done
from pettingzoo import AECEnv
# from pettingzoo import ParallelEnv


def call_reset(parcel: dict, *, env: AECEnv, save_obs_as="obs"):
  env.reset()
  parcel.update(
    zip([save_obs_as, 'reward', 'terminated', 'truncated', 'info'], env.last())
  )
  parcel['agent'] = env.agent_selection


def call_step(parcel: dict, *, env: AECEnv, save_obs_as="obs"): 
  action = parcel['action']
  env.step(action)
  parcel.update(
    zip([save_obs_as, 'reward', 'terminated', 'truncated', 'info'], env.last())
  )
  parcel['agent'] = env.agent_selection


def call_render(parcel: dict, *, env: AECEnv):
  env.render()


def check_done(parcel: dict):
  if parcel.get('terminated', False) or parcel.get('truncated', False):
    raise Done
