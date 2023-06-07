# turingpoint

Turing point is a Reinforcement Learning (RL) library. It adds the missing duct tape.
It allows for multiple (hetrogenous) agents seamlessly. Per-agent partial observation is natural with Turing point.
Different agents can act in differnet frequencies.
You may opt to continue using also the environment and the agent libraries that you're currently using, for the such as *Gym/Gymnasium*, *Stable-Baselines3*, *Tianshou*, *RLLib*, etc.
Turing point integrates easily with existing RL libraries and your own custom code.
Integration of RL agents in the target applications should be significantly easier with Turing point.

The main concept in Turing point is that there are multiple participants and each gets its turn.
The participants communicate by a parcel that is passed among them. The agent and the environment are both participants in that sense. No more confusion which of those should call which. Reward's logic, for example,
can be addressed where you believe is most suitable.

Turing point may be helpful with parallel or distributed training, yet Turing point does not address those explicitly. On the contrary; with Turing point the flow is sequential among the participants. As far as we can tell Turing point at least does not hinder the use of parallel and/or distributed training.

Participants can be added and/or removed dynamically (ex. a new "monster" enters or then "disappears").

## Participant

Every component in the main "episodic" loop needs to implement the ```Participant``` protocol. This basically means that it should be a callable that receives a "parcel". The parcel is a dict, where one can expect to find things like *action*, *reward*, *observation*, etc.  

In the example below, we define a "participant" that increments its own "counter" everytime it is being called.

``` py
def participant1(parcel: dict) -> None:
    parcel['participant1'] = parcel.get('participant1', 0) + 1
```

Note the the *parcel* dict above could have potentailly have the *obs* key.
It is just that *participant1* did not check for it.
On the other hand, if the key *obs* was there before the call, it should still be there in the *parcel*.

The *parcel* is passed among the participants, each has an option to examine the contents and to add/modify/remove entries.

The following *participant* (implemented in *utils.py*) can be used to collect rollouts:

``` py
class Collector:
  """Simple "participant" that records specific values from the parcel.
  """

  def __init__(self, keys_to_collect = ['obs', 'action', 'new_obs', 'reward', 'done']):
    self._keys_to_collect = keys_to_collect
    self._entries = [] # TODO: potentially replace with dqueue

  def __call__(self, parcel: dict) -> None:
    new_entry = {k: parcel[k] for k in self._keys_to_collect}
    self._entries.append(new_entry)

  def get_entries(self) -> Generator[dict, None, None]:
    yield from self._entries

  def clear_entries(self) -> None:
    self._entries.clear()
```

## Assembly

Assembly in *turingpoint* is an abstract class that enables an "episodic" loop. One needs to provide implementation for the following member functions: *create_initial_parcel*, and *participants*.

*get_participants* is a callable that returns an iterable or an iterator for the participants. You can imagine a simple setting where you have a Gymnasium environment and a Stable-Baselines3 agent. The iterator shall return once the agent, and then the environment, and so force.
In the example, both the environment and the agent needs to be wrapped as a participant. That is to adapt the *step* and *predict* API accordingly.

``` py linenums="1"
import functools
import itertools

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.sb3_utils as tp_sb3_utils
import turingpoint.utils as tp_utils
import turingpoint as tp


def evaluate(env, agent, num_episodes: int) -> float:
    total_reward = 0

    def bookkeeping(parcel: dict) -> None:
        nonlocal total_reward

        reward = parcel['reward']
        total_reward += reward

    def get_participants():
      yield functools.partial(tp_gym_utils.call_reset, env=env)
      yield from itertools.cycle([
          functools.partial(
            tp_sb3_utils.call_predict,
            agent=agent, deterministic=True
          ),
          functools.partial(tp_gym_utils.call_step, env=env),
          bookkeeping,
          tp_gym_utils.check_done
      ]) 

    assembly = tp.Assembly(get_participants)
      
    for _ in range(num_episodes):
        _ = assembly.launch()

    return total_reward / num_episodes
```

Note that *launch* is being called as many time as needed (as many episodes as needed).
A new parcel is created (behind the scenes) for every call of *launch*.
Also note that *launch* is returning the parcel as is at the end of the episode, yet we've decided above to discard it (discard the parcel and not look into it).
