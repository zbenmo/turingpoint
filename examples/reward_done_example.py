from typing import List, Generator
from turingpoint import Assembly, Participant
from turingpoint.utils import print_parcel


def reward_done_example():
  position: int = 5
  total_reward: int = 0
  episode = None

  def set_observation(parcel: dict):
    """
    helper to avoid repetition
    """
    parcel['obs'] = position

  class MyAssembly(Assembly):
    def __init__(self, participants_list: List[Participant]):
      self.participants_list = participants_list

    def create_initial_parcel(self) -> dict:
      """
      The initial parcel, the first observation from the environment.
      """
      nonlocal episode

      parcel = {}
      set_observation(parcel)
      episode = [{'obs': parcel['obs']}] # this will be the start of the episode
      return parcel

    def participants(self) -> Generator[Participant, None, None]:
      done = False

      def check_done(parcel: dict) -> None:
        "a helper participant for checking for 'done' in the parcel"

        nonlocal done
        done = parcel.get('done', False)

      while not done:
        yield from self.participants_list
        yield check_done

  def agent(parcel: dict) -> None:
    """
    participant - the agent
    """
    nonlocal episode

    obs = parcel['obs']
    assert obs > 0
    action = 'go_left'
    parcel['action'] = action

  def environment(parcel: dict) -> None:
    """
    participant - the environment
    """
    nonlocal position

    action = parcel['action']
    if action == 'go_left':
      position = position - 1
      set_observation(parcel)
    if position < 1:
      parcel['done'] = True
      reward = -10
    else:
      reward = +1
    parcel['reward'] = reward

  def monitor_episode(parcel: dict) -> None:
    nonlocal total_reward

    total_reward += parcel['reward']
    episode.append({
      key: parcel[key] for key in ['action', 'reward', 'obs']
    })

  assembly = MyAssembly([print_parcel, agent, environment, monitor_episode, print_parcel])

  assembly.launch()

  print(f'The agent collected {total_reward} total reward.')
  print(f"The episode was {episode}")


if __name__ == "__main__":
  reward_done_example()
