# turingpoint

Turing point is a Reinforcement Learning (RL) library.
It allows for multiple (hetrogenous) agents seamlessly. Per-agent partial observation is natural with Turing point.
Different agents can act in differnet frequencies.
You may opt to continue using also the libraries that you're currently using, such as Gym, Stable-Baselines3, RLLib, etc.
Turing point integrates easily with existing RL libraries and your own custom code.
Integration of RL agents in the target applications should be significantly easier with Turing point.

Consider a Gym/SB3 training setting:

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
the software that is used to provision those environment/agent(s) settings.

Let's see how above can be described with Turing point:

```python
import gym

from stable_baselines3 import A2C

import turingpoint as tp


class MyAgent(tp.Agent):
    def __init__(self, env, learn_initialy_total_timesteps=10_000, learn_online=False):
        super().__init__(self)
        self.model = A2C("MlpPolicy", env, verbose=1)
        self.model.learn(learn_initialy_total_timesteps=learn_initialy_total_timesteps)
        self.learn_online = learn_online
        self.memory = {
            ...
        }

    def _being(self) -> Generator[str, dict, None]:
        obs = yield None
        while True:
            if self.learn_online:
                pass # TODO:
            if obs.get('done', False):
                break
            action, _state = self.model.predict(obs.get('obs'), deterministic=True) 
            obs = yield action

    def reset(self):
        super().reset()
        self.memory = {
            ...
        }


env = gym.make("CartPole-v1")
agent = MyAgent(env)

env_obs = env.reset()
obs = {
    'env_obs': env_obs # in this case
}
for i in range(1000):
    action = agent.react(obs)
    env_obs, reward, done, info = env.step(action)
    obs = {
        'env_obs': env_obs, # in this case
        'reward': reward, # in this case
    }
    if done:
        obs['done'] = True
    env.render()
    if done:
      env_obs = env.reset()
      agent.reset()
```

What did we gain and was it worth the extra coding? Let's add to the environment a second agent, wind, or maybe it is part of the augmented environment, does not really matter. Let's just add it.


Below is wrong in many aspects. For example how will the environment learn that the wind influenced the angle etc. ### TODO!!:
====

```python
import gym

from stable_baselines3 import A2C

import turingpoint as tp


class MyAgent(tp.Agent):
    def __init__(self, env, learn_initialy_total_timesteps=10_000, learn_online=False):
        super().__init__(self)
        self.model = A2C("MlpPolicy", env, verbose=1)
        self.model.learn(learn_initialy_total_timesteps=10_000)
        self.learn_online = learn_online
        self.memory = {
            ...
        }

    def _being(self) -> Generator[str, dict, None]:
        obs = yield None
        while True:
            if self.learn_online:
                pass # TODO:
            if obs.get('done', False):
                break
            action, _state = self.model.predict(obs.get('obs'), deterministic=True) 
            obs = yield action

    def reset(self):
        super().reset()
        self.memory = {
            ...
        }


class MyWind(tp.Agent):
    def _being(self) -> Generator[str, dict, None]:
        obs = yield None
        while True:
            if obs.get('done', False):
                break
            action = "blow left" if random() < 0.5 else "blow right"
            obs = yield action


env = gym.make("CartPole-v1")
agent = MyAgent(env, learn_initialy_total_timesteps=0, learn_online=True)
wind = MyWind()

env_obs = env.reset()
obs = {
    'env_obs': env_obs, # in this case
    'wind': wind.react({})
}
for i in range(1000):
    action = agent.react(obs)
    env_obs, reward, done, info = env.step(action)
    obs = {
        'env_obs': env_obs, # in this case
        'reward': reward, # in this case
    }
    if done:
        obs['done'] = True
    env.render()
    if done:
      env_obs = env.reset()
      my_agent.reset()
wind.react({'done': True})
```

