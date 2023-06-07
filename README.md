# turingpoint

Turing point is a Reinforcement Learning (RL) library. It adds the missing duct tape.
It allows for multiple (hetrogenous) agents seamlessly. Per-agent partial observation is natural with Turing point.
Different agents can act in differnet frequencies.
You may opt to continue using also the environment and the agent libraries that you're currently using, for the such as Gym/Gymnasium, Stable-Baselines3, Tianshou, RLLib, etc.
Turing point integrates easily with existing RL libraries and your own custom code.
Integration of RL agents in the target applications should be significantly easier with Turing point.

The main concept in Turing point is that there are multiple participants and each gets its turn.
The participants communicate by a parcel that is passed among them. The agent and the environment are both participants in that sense. No more confusion which of those should call which. Reward's logic, for example,
can be addressed where you believe is most suitable.

Turing point may be helpful with parallel or distributed training, yet Turing point does not address those explicitly. On the contrary; with Turing point the flow is sequential among the participants. As far as we can tell Turing point at least does not hinder the use of parallel and / or distributed training.

Participants can be added and / or removed dynamically (ex. a new "monster" enters or then "disappears").

Consider a Gym/SB3 training realm:

```python
import gym

from stable_baselines3 import A2C

# Creating the specific Gym environment.
env = gym.make("CartPole-v1")

# An agent is created, it is injected with the environment.
# The agent probably makes a copy of the passed environment, wraps it etc.
model = A2C("MlpPolicy", env, verbose=1)

# The agent is trained against its environment.
# We can assume what is happening there (obs, action, reward, obs, ..), yet it is not explicit.
model.learn(total_timesteps=10_000)

# we now evaluate the performance of our agent with the help of the environment that the agent maintains.
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    # The parameter for predict is the observation,
    #  which is good as our application (ex. an actual cartpole robot) can indeed provide such observations and use the return action.
    # Note: the action space as well as the observation space are defined in the environment.
    # Also note. The environment is aware of the agent. This is how the environment was designed.
    # The action space of the agent is coded in the environment.
    # The observation space is intended for the agent and reflects probably also what the agent should know about itself.
    # The _state output is related to RNNs, AFAIK.
    action, _state = model.predict(obs, deterministic=True)
    # Here the reward, done, and info outputs are just for our evaluation.
    # Mainly what is happening here is that the environment moves to a new state.
    # The reward and done flag, are specific to the agent.
    # If there are other entities in the environments, those may continue to live also after done=True and may not care (directly) about this specific reward.
    obs, reward, done, info = vec_env.step(action)
    # We render here. We did not render during the training(learn) which probably makes sense performace wise.
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

# Observation: we reset the environment. The model is supposed to be memory-less (MDP assumption). 
```

In the comments above, we've tried to give the intuition why some additional thinking is needed about
the software that is used to provision those environment / agent(s) realms.

Let's see how above can be described with Turing point:

```python
...
import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.sb3_utils as tp_sb3_utils
import turingpoint.utils as tp_utils
import turingpoint as tp


def evaluate(env, agent, num_episodes: int) -> float:

  rewards_collector = tp_utils.Collector(['reward'])

  def get_participants():
    yield functools.partial(tp_gym_utils.call_reset, env=env)
    yield from itertools.cycle([
        functools.partial(tp_sb3_utils.call_predict, agent=agent, deterministic=True),
        functools.partial(tp_gym_utils.call_step, env=env),
        rewards_collector,
        tp_gym_utils.check_done
    ]) 

  evaluate_assembly = tp.Assembly(get_participants)

  for _ in range(num_episodes):
    _ = evaluate_assembly.launch()
    # Note that we don't clear the rewards in 'rewards_collector', and so we continue to collect.

  total_reward = sum(x['reward'] for x in rewards_collector.get_entries())

  return total_reward / num_episodes

..

def main():

  random.seed(1)
  np.random.seed(1)
  torch.manual_seed(1)

  env = gym.make('CartPole-v1')

  env.reset(seed=1)

  agent = PPO(MlpPolicy, env, verbose=0) # use verbose=1 for debugging

  mean_reward_before_train = evaluate(env, agent, 100)
  print("before training")
  print(f'{mean_reward_before_train=}')

..
```

What did we gain and was it worth the extra coding? Let's add to the environment a second agent, wind, or maybe it is part of the augmented environment, does not really matter. Let's just add it.

```python
..

def wind(parcel: dict) -> None:
    action_wind = "blow left" if random() < 0.5 else "blow right"
    parcel['action_wind'] = action_wind


def wind_impact(parcel: dict) -> None:
    action_wind = parcel['action_wind']
    # We'll modify the action of the agent, given the wind,
    # as we don't have here access to the state of the environment.
    parcel['action'] = ...


def evaluate(env, agent, num_episodes: int) -> float:

  rewards_collector = tp_utils.Collector(['reward'])

  def get_participants():
    yield functools.partial(tp_gym_utils.call_reset, env=env)
    yield from itertools.cycle([
        functools.partial(tp_sb3_utils.call_predict, agent=agent, deterministic=True),
        wind,
        wind_impact,
        functools.partial(tp_gym_utils.call_step, env=env),
        rewards_collector,
        tp_gym_utils.check_done
    ]) 

  evaluate_assembly = tp.Assembly(get_participants)

  for _ in range(num_episodes):
    _ = evaluate_assembly.launch()
    # Note that we don't clear the rewards in 'rewards_collector', and so we continue to collect.

  total_reward = sum(x['reward'] for x in rewards_collector.get_entries())

  return total_reward / num_episodes
```

To install use for example:

```
pip install turingpoint
```

The examples are found in the homepage (github) under the 'examples' folder.
