from typing import Tuple, Optional
from dataclasses import dataclass
# import dataclasses
from turingpoint.assembly import Assembly
from turingpoint.utils import (
  generator_from_list,
  Event
)


# def two_episodes_example():
#   position: int = 5

#   def get_state() -> int:
#     return position

#   def scatter_observations(state: int) -> str:
#     print(f'position={state}')
#     return "go_left"

#   def apply_actions(actions: str) -> bool:
#     nonlocal position

#     print(f'action={actions}')
#     done = False
#     if actions == 'go_left':
#       position = position - 1
#     if position < 1:
#       done = True
#     return done

#   def observe_results(results: bool) -> bool:
#     return results

#   environment = Assemble(
#     get_state=get_state,
#     scatter_observations=scatter_observations,
#     apply_actions=apply_actions,
#     observe_results=observe_results
#   )

#   for episode in range(2):
#     print(f'{episode=}')
#     print("--")
#     environment.launch()
#     print()
#     position = 5


# def reward_done_example():
#   position: int = 5
#   total_reward: int = 0
#   episode = []

#   def get_state() -> int:
#     return position

#   def scatter_observations(state: int) -> str:
#     print(f'position={state}')
#     if len(episode) < 1:
#       episode.append({
#         'obs': state
#       })
#     else:
#       episode[-1]['obs'] = state
#     return "go_left"

#   def apply_actions(actions: str) -> Tuple[bool, int]:
#     nonlocal position

#     print(f'action={actions}')
#     episode.append({
#       'action': actions
#     })
#     done = False
#     if actions == 'go_left':
#       position = position - 1
#     if position < 1:
#       done = True
#       reward = -10
#     else:
#       reward = +1
#     return done, reward

#   def observe_results(results: Tuple[bool, int]) -> bool:
#     nonlocal total_reward

#     done, reward = results
#     episode.append({
#       'reward': reward
#     })
#     total_reward += reward
#     return done

#   environment = Assemble(
#     get_state=get_state,
#     scatter_observations=scatter_observations,
#     apply_actions=apply_actions,
#     observe_results=observe_results
#   )

#   environment.launch()

#   print(f'The agent collected {total_reward} total reward.')
#   print(f"Judging from the agent's recollection, the episode is {episode}")


def main():
  pass
  # print("two_episodes_example")
  # print("--------------------")
  # two_episodes_example()
  # print()
  # print("reward_done_example")
  # print("--------------------")
  # reward_done_example()
  # print()


if __name__ == "__main__":
  main()
