# turingpoint

Turing point is a Reinforcement Learning (RL) library.
It allows for multiple (hetrogenous) agents seamlessly. Per-agent partial observation is natural with Turing point.
Different agents can act in differnet frequencies.
You may opt to continue using also the libraries that you're currently using, such as Gym, Stable-Baselines3, RLLib, etc.
Turing point integrates easily with existing RL libraries and your own custom code.
Integration of RL agents in the target applications should be significantly easier with Turing point.

The main concept in Turing point is that there are multiple participants and each gets its turn.
The participants communicate by a parcel that is passed among them. The agent and the environment are both participants in that sense. No more confusion which of those should call which. Reward's logic, for example,
can be addressed where you believe is most suitable.

Turing point may be helpful with parallel or distributed training, yet it does not address those implicitly. On the contrary; with Turing point the flow is sequential between the participants. As far as we can tell Turing point at least does not hinder the use of parallel and / or parallel training.

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
    # The _state output is a bit suspicious. It is here probably as the model also predicts the state.
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
import gym

from stable_baselines3 import A2C

from turingpoint.gym_utils import (
  AgentParticipant,
  EnvironmentParticipant,
  RenderParticipant,
  GymAssembly
)


# Creating the specific Gym environment.
env = gym.make("CartPole-v1")

# An agent is created, it is injected with the environment.
# The agent probably makes a copy of the passed environment, wraps it etc.
model = A2C("MlpPolicy", env, verbose=1)

# The agent is trained against its environment.
# We can assume what is happening there (obs, action, reward, obs, ..), yet it is not explicit.
model.learn(total_timesteps=10_000)

# above starts the same

# now ..

vec_env = model.get_env()
assembly = GymAssembly(vec_env, [
    AgentParticipant(agent),
    EnvironmentParticipant(vec_env),
    RenderParticipant(vec_env)
])

for i in range(1000):
    assembly.launch()
```

What did we gain and was it worth the extra coding? Let's add to the environment a second agent, wind, or maybe it is part of the augmented environment, does not really matter. Let's just add it.

```python
import gym

from stable_baselines3 import A2C

from turingpoint.gym_utils import (
  AgentParticipant,
  EnvironmentParticipant,
  RenderParticipant,
  GymAssembly
)


# Creating the specific Gym environment.
env = gym.make("CartPole-v1")

# An agent is created, it is injected with the environment.
# The agent probably makes a copy of the passed environment, wraps it etc.
model = A2C("MlpPolicy", env, verbose=1)

# The agent is trained against its environment.
# We can assume what is happening there (obs, action, reward, obs, ..), yet it is not explicit.
model.learn(total_timesteps=10_000)

def wind(parcel: dict) -> None:
    action_wind = "blow left" if random() < 0.5 else "blow right"
    parcel['action_wind'] = action_wind

def wind_impact(parcel: dict) -> None:
    action_wind = parcel['action_wind']
    # We'll modify the action of the agent, given the wind,
    # as we don't have here access to the state of the environment.
    parcel['action'] = ...

vec_env = model.get_env()
assembly = GymAssembly(vec_env, [
    AgentParticipant(agent),
    wind,
    wind_impact,
    EnvironmentParticipant(vec_env),
    RenderParticipant(vec_env)
])

for i in range(1000):
    assembly.launch()
```

To install use for example:

```
pip install turingpoint
```

The examples are found in the homepage (github) under the 'examples' folder.
