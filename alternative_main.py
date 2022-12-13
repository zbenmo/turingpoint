from typing import Tuple, Optional
from dataclasses import dataclass
from turingpoint.environment import Environment


def one_agent_example():
  position: int = 5

  def get_state() -> int:
    return position

  def scatter_observations(state: int) -> str:
    print(f'position={state}')
    return "go_left"

  def apply_actions(actions: str) -> bool:
    nonlocal position

    print(f'action={actions}')
    done = False
    if actions == 'go_left':
      position = position - 1
    if position < 1:
      done = True
    return done

  def observe_results(results: bool) -> bool:
    return results

  environment = Environment(
    get_state=get_state,
    scatter_observations=scatter_observations,
    apply_actions=apply_actions,
    observe_results=observe_results
  )

  environment.launch()


def two_agents_example():

  @dataclass
  class WorldState:
    agent1: bool
    position_agent1: int
    position_agent2: int

  state = WorldState(True, 5, 1)

  def get_state() -> WorldState:
    return state

  def agent1_logic(position1: int) -> str:
    return "go_left"

  def agent2_logic(you: int, pray: Optional[int] = None) -> str:
    if pray is None:
      return "go_left"
    else:
      return "go_right"

  def scatter_observations(state: WorldState) -> Tuple[str, str]:
    if state.agent1:
      print(f'position1={state.position_agent1}')
    print(f'position2={state.position_agent2}')
    if state.agent1:
      agent1_action = agent1_logic(state.position_agent1)
      agent2_action = agent2_logic(state.position_agent2, state.position_agent1)
    else:
      agent1_action = None
      agent2_action = agent2_logic(state.position_agent2)
    return (agent1_action, agent2_action)

  def apply_actions(actions: Tuple[str, str]) -> bool:
    nonlocal state
    
    print(f'actions={actions}')
    done = False
    if actions[0] == 'go_left':
      state.position_agent1 += -1
    if actions[1] == 'go_right':
      state.position_agent2 += +1
    elif actions[1] == 'go_left':
      state.position_agent2 += -1
    if state.agent1:
      if state.position_agent1 <= state.position_agent2:
        state.agent1 = False
    if state.position_agent2 < 0:
      done = True
    return done # Here we'll only report on the end of the whole episode for agent2, but we could've report about agent1 also and more.

  def observe_results(results: bool) -> bool:
    return results

  environment = Environment(
    get_state=get_state,
    scatter_observations=scatter_observations,
    apply_actions=apply_actions,
    observe_results=observe_results
  )

  environment.launch()


def two_episodes_example():
  position: int = 5

  def get_state() -> int:
    return position

  def scatter_observations(state: int) -> str:
    print(f'position={state}')
    return "go_left"

  def apply_actions(actions: str) -> bool:
    nonlocal position

    print(f'action={actions}')
    done = False
    if actions == 'go_left':
      position = position - 1
    if position < 1:
      done = True
    return done

  def observe_results(results: bool) -> bool:
    return results

  environment = Environment(
    get_state=get_state,
    scatter_observations=scatter_observations,
    apply_actions=apply_actions,
    observe_results=observe_results
  )

  for episode in range(2):
    print(f'{episode=}')
    print("--")
    environment.launch()
    print()
    position = 5


def reward_done_example():
  position: int = 5
  total_reward: int = 0
  episode = []

  def get_state() -> int:
    return position

  def scatter_observations(state: int) -> str:
    print(f'position={state}')
    if len(episode) < 1:
      episode.append({
        'obs': state
      })
    else:
      episode[-1]['obs'] = state
    return "go_left"

  def apply_actions(actions: str) -> Tuple[bool, int]:
    nonlocal position

    print(f'action={actions}')
    episode.append({
      'action': actions
    })
    done = False
    if actions == 'go_left':
      position = position - 1
    if position < 1:
      done = True
      reward = -10
    else:
      reward = +1
    return done, reward

  def observe_results(results: Tuple[bool, int]) -> bool:
    nonlocal total_reward

    done, reward = results
    episode.append({
      'reward': reward
    })
    total_reward += reward
    return done

  environment = Environment(
    get_state=get_state,
    scatter_observations=scatter_observations,
    apply_actions=apply_actions,
    observe_results=observe_results
  )

  environment.launch()

  print(f'The agent collected {total_reward} total reward.')
  print(f"Judging from the agent's recollection, the episode is {episode}")


def main():
  print("one_agent_example")
  print("-----------------")
  one_agent_example()
  print()
  print("two_agents_example")
  print("-----------------")
  two_agents_example()
  print()
  print("two_episodes_example")
  print("--------------------")
  two_episodes_example()
  print()
  print("reward_done_example")
  print("--------------------")
  reward_done_example()
  print()


if __name__ == "__main__":
  main()
