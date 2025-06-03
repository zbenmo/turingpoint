import datetime
import functools
import itertools
import random
from typing import Dict, List
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint as tp


gym_environment = "LunarLander-v3"


# copying stuff from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class StateToActionLogits(nn.Module):
	def __init__(self, in_features, out_features, *args, **kwargs):
		super().__init__(*args, **kwargs)
		hidden_features = 64
		self.net = nn.Sequential(
			layer_init(nn.Linear(in_features, hidden_features)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_features, hidden_features)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_features, out_features), std=0.01),
		)

	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		"""obs -> actions (logits)"""

		return self.net(obs)


class StateToValue(nn.Module):
	def __init__(self, in_features, *args, **kwargs):
		super().__init__(*args, **kwargs)
		hidden_features = 64
		self.net = nn.Sequential(
            layer_init(nn.Linear(in_features, hidden_features)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_features, hidden_features)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_features, 1), std=1.0),
        )

	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		"""obs -> value (a regression)"""

		return self.net(obs)
	

def get_action(parcel: Dict, *, agent: StateToActionLogits):
	"""Picks a random action based on the probabilities that the agent assigns.
	Just needs to account for the fact the the agent actually returns logits rather than probabilities.
	"""
	obs = parcel['obs']
	with torch.no_grad():
		logits = agent(torch.tensor(obs).unsqueeze(0))
		exps = torch.exp(logits.squeeze(0))
		probs = exps / exps.sum()
		action = torch.multinomial(probs, 1).item()
	parcel['action'] = action
	parcel['prob'] = probs[action] # may be useful for the training (note: still a tensor)


def evaluate(env, agent, num_episodes: int) -> float:

	rewards_collector = tp_utils.Collector(['reward'])

	def get_participants():
		yield functools.partial(tp_gym_utils.call_reset, env=env)
		yield from itertools.cycle([
				functools.partial(get_action, agent=agent),
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


def collect_episodes(env, agent, num_episodes=40) -> List[List[Dict]]:

	collector = tp_utils.Collector(['obs', 'action', 'prob', 'reward'])

	def get_episode_participants():
		yield functools.partial(tp_gym_utils.call_reset, env=env)
		yield from itertools.cycle([
				functools.partial(get_action, agent=agent),
				functools.partial(tp_gym_utils.call_step, env=env),
				collector,
				tp_gym_utils.check_done
		])

	episodes_assembly = tp.Assembly(get_episode_participants)

	episodes = [] # it will be a list of lists of dictionaries
	for _ in range(num_episodes):
		_ = episodes_assembly.launch()
		episode = list(collector.get_entries())
		collector.clear_entries()
		episodes.append(episode)
	return episodes


# Taken from CoPilot 
def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards (np.array): Rewards from the environment.
        values (np.array): Value function estimates.
        gamma (float): Discount factor.
        lambda_ (float): GAE coefficient.

    Returns:
        np.array: Computed advantage estimates.
    """
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = delta + gamma * lambda_ * last_advantage
        last_advantage = advantages[t]
    
    return advantages

# # Example usage:
# rewards = np.array([1, 2, 3, 4, 5])
# values = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5])  # Last value is estimated for bootstrapping
# gamma = 0.99
# lambda_ = 0.95

# advantages = compute_gae(rewards, values, gamma, lambda_)


def train(env, agent, critic, total_timesteps):
	causality_to_be_accounted_for = True
	normilize_the_rewards = True
	discount = 0.99 # gamma
	gae = 0.95
		
	optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

	critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

	writer = SummaryWriter(
		f"runs/{gym_environment}_ppo_{datetime.datetime.now().strftime('%I_%M%p_on_%B_%d_%Y')}_{causality_to_be_accounted_for=}_{normilize_the_rewards=}"
	) # TensorBoard

	timesteps = 0
	with tqdm(total=total_timesteps, desc="training steps") as pbar:
		while timesteps < total_timesteps:

			critic_optimizer.zero_grad()

			optimizer.zero_grad()

			episodes = collect_episodes(env, agent)

			# Now learn from above episodes

			# disclaimer: a lot more of vectorization can be done here.
			# potentially a usage of pandas/polars, numpy, or torch. 
			# I'm KISSing it here.

			obs_batch = []
			action_batch = []
			probs_batch = []
			values_batch = []
			advantage_batch = []

			total_rewards = []

			for episode in episodes:
				obs, actions, rewards, probs = zip(*((e['obs'], e['action'], e['reward'], e['prob']) for e in episode))
				obs_batch.extend(obs)
				action_batch.extend(actions)
				probs_batch.extend(probs)
				values = critic(torch.tensor(np.array(obs))).squeeze(-1).cpu().tolist()
				values.append(0.)
				values_batch.extend(r + discount * v for r, v in zip(rewards, values[1:]))
				advantages = compute_gae(rewards, values, gamma=discount, lambda_=gae)
				advantage_batch.extend(advantages)
				total_reward = sum(rewards) # TODO: discounted reward?
				total_rewards.append(total_reward)

			obs_tensor = torch.tensor(np.array(obs_batch))
			actions_tensor = torch.tensor(action_batch)
			advantages_tensor = torch.tensor(advantage_batch)
			probs_batch_tensor = torch.tensor(probs_batch)
			values_tensor = torch.tensor(values_batch, dtype=torch.float32)

			assert len(probs_batch_tensor) == len(advantages_tensor), f'{len(probs_batch_tensor)=}, {len(advantages_tensor)=}'

			if normilize_the_rewards:
				advantages_tensor_mean = advantages_tensor.mean()
				advantages_tensor = (advantages_tensor - advantages_tensor_mean) / (advantages_tensor.std() + 1e-5)

			timesteps += len(probs_batch_tensor)

			logits = agent(obs_tensor)

			exps = torch.exp(logits)
			probs = exps / exps.sum(dim=-1, keepdim=True)
			probs = probs.gather(dim=1, index=actions_tensor.view(-1, 1)).squeeze(-1)

			assert probs.shape == probs_batch_tensor.shape, f'{probs.shape=}, {probs_batch_tensor.shape=}'

			r = probs / probs_batch_tensor
			loss1 = r * advantages_tensor
			clip_coef = 0.25
			loss2 = torch.clip(r, 1. - clip_coef, 1. + clip_coef) * advantages_tensor

			loss = -torch.min(loss1, loss2).mean() # we want to maximize

			loss.backward()

			optimizer.step()

			# optimize the critic

			critic_loss = F.mse_loss(critic(obs_tensor).squeeze(-1), values_tensor)

			critic_loss.backward()

			critic_optimizer.step()

			#

			writer.add_scalar("Mean Rewards/train", np.mean(total_rewards), timesteps)
			writer.add_scalar("Loss/train", loss, timesteps)
			writer.add_scalar("Loss Critic/train", critic_loss, timesteps)

			pbar.update(len(advantages_tensor))

	writer.flush()


def main():

	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(1)

	env = gym.make(gym_environment)

	env.reset(seed=1)

	# state and obs/observations are used in this example interchangably.

	state_space = env.observation_space.shape[0]
	action_space = env.action_space.n

	agent = StateToActionLogits(state_space, action_space)
	critic = StateToValue(state_space)

	mean_reward_before_train = evaluate(env, agent, 100)
	print("before training")
	print(f'{mean_reward_before_train=}')

	train(env, agent, critic, total_timesteps=2_000_000)

	mean_reward_after_train = evaluate(env, agent, 100)
	print("after training")
	print(f'{mean_reward_after_train=}')


if __name__ == "__main__":
	main()
