import datetime
import functools
import itertools
from itertools import islice
import random
from typing import Dict, List
import numpy as np
import optuna
import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as dist
from tqdm import tqdm

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint as tp
import turingpoint.torch_utils as tp_torch_utils


gym_environment = "Taxi-v3"

embedding_dim = 4

class StateToActionLogits(nn.Module):
	def __init__(self, in_features, out_features, *args, **kwargs):
		super().__init__(*args, **kwargs)
		hidden_features = 64
		self.net = nn.Sequential(
			nn.Embedding(in_features, embedding_dim),
			tp_torch_utils.layer_init(nn.Linear(embedding_dim, hidden_features)),
			nn.Tanh(),
			tp_torch_utils.layer_init(nn.Linear(hidden_features, hidden_features)),
			nn.Tanh(),
			tp_torch_utils.layer_init(nn.Linear(hidden_features, out_features), std=0.01),
		)

	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		"""obs -> actions (logits)"""

		return self.net(obs)


class StateToValue(nn.Module):
	def __init__(self, in_features, *args, **kwargs):
		super().__init__(*args, **kwargs)
		hidden_features = 64
		self.net = nn.Sequential(
			nn.Embedding(in_features, embedding_dim),
            tp_torch_utils.layer_init(nn.Linear(embedding_dim, hidden_features)),
            nn.Tanh(),
            tp_torch_utils.layer_init(nn.Linear(hidden_features, hidden_features)),
            nn.Tanh(),
            tp_torch_utils.layer_init(nn.Linear(hidden_features, 1), std=1.0),
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
		action_dist = dist.Categorical(logits=logits)
		action = action_dist.sample()
		parcel['action'] = action.item()
		parcel['log_prob'] = action_dist.log_prob(action) # may be useful for the training (note: still a tensor)


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

	collector = tp_utils.Collector(['obs', 'action', 'log_prob', 'reward'])

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


def train(optuna_trial, env, agent, critic, total_timesteps):
	causality_to_be_accounted_for = True
	normilize_the_rewards = True
	discount = optuna_trial.suggest_float('discount', 0.99, 0.99) # gamma
	gae = optuna_trial.suggest_float('gae_lambda', 0.95, 0.95) # lambda
	clip_coef = optuna_trial.suggest_float('clip_coef', 0.2, 0.2)
	actor_lr = optuna_trial.suggest_float('actor_lr', 1e-4, 1e-4)
	critic_lr = optuna_trial.suggest_float('critic_lr', 1e-4, 1e-4)
		
	optimizer = torch.optim.Adam(agent.parameters(), lr=actor_lr)

	critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

	writer = SummaryWriter(
		f"runs/{gym_environment}_ddqn_{optuna_trial.datetime_start.strftime('%Y_%B_%d__%H_%M%p')}_study_{optuna_trial.study.study_name}_trial_no_{optuna_trial.number}"
	) # TensorBoard

	timesteps = 0
	with tqdm(total=total_timesteps, desc="training steps") as pbar:
		while timesteps < total_timesteps:

			episodes = collect_episodes(env, agent)

			# Now learn from above episodes

			# disclaimer: a lot more of vectorization can be done here.
			# potentially a usage of pandas/polars, numpy, or torch. 
			# I'm KISSing it here.

			obs_batch = []
			action_batch = []
			log_probs_batch = []
			values_batch = []
			advantage_batch = []

			total_rewards = []

			for episode in episodes:
				obs, actions, rewards, log_probs = zip(*((e['obs'], e['action'], e['reward'], e['log_prob']) for e in episode))
				obs_batch.extend(obs)
				action_batch.extend(actions)
				log_probs_batch.extend(log_probs)
				values = critic(torch.tensor(np.array(obs))).squeeze(-1).cpu().tolist()
				values.append(0.)
				values_batch.extend(r + discount * v for r, v in zip(rewards, values[1:]))
				advantages = tp_utils.compute_gae(rewards, values, gamma=discount, lambda_=gae)
				advantage_batch.extend(advantages)
				total_reward = sum(rewards) # TODO: discounted reward?
				total_rewards.append(total_reward)

			obs_tensor = torch.tensor(np.array(obs_batch))
			actions_tensor = torch.tensor(action_batch)
			advantages_tensor = torch.tensor(advantage_batch).to(torch.float32)
			log_probs_batch_tensor = torch.tensor(log_probs_batch)
			values_tensor = torch.tensor(values_batch, dtype=torch.float32)

			assert len(log_probs_batch_tensor) == len(advantages_tensor), f'{len(log_probs_batch_tensor)=}, {len(advantages_tensor)=}'

			if normilize_the_rewards:
				advantages_tensor_mean = advantages_tensor.mean()
				advantages_tensor = (advantages_tensor - advantages_tensor_mean) / (advantages_tensor.std() + 1e-5)

			timesteps += len(log_probs_batch_tensor)

			ds = TensorDataset(obs_tensor, advantages_tensor, values_tensor, actions_tensor, log_probs_batch_tensor)
			dl = DataLoader(ds, batch_size=1024, shuffle=True)

			for epoch in range(4):
				for o, adv, v, act, l_p in islice(dl, None):

					critic_optimizer.zero_grad()

					optimizer.zero_grad()

					logits = agent(o)
					action_dist = dist.Categorical(logits=logits)
					log_probs = action_dist.log_prob(act)

					# assert probs.shape == probs_batch_tensor.shape, f'{probs.shape=}, {probs_batch_tensor.shape=}'

					ratio = (log_probs - l_p).exp()
					loss1 = ratio * adv
					loss2 = torch.clip(ratio, 1. - clip_coef, 1. + clip_coef) * adv

					loss = -torch.min(loss1, loss2).mean() # we want to maximize

					loss.backward()

					optimizer.step()

					# optimize the critic

					critic_loss = F.mse_loss(critic(o).squeeze(-1), v)

					critic_loss.backward()

					critic_optimizer.step()

					#

					writer.add_scalar("Loss/train", loss, timesteps)
					writer.add_scalar("Loss Critic/train", critic_loss, timesteps)

			writer.add_scalar("Mean Rewards/train", np.mean(total_rewards), timesteps)

			pbar.update(len(advantages_tensor))

	writer.flush()


def optuna_objective(optuna_trial):

	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(1)

	env = gym.make(gym_environment)

	env.reset(seed=1)

	# state and obs/observations are used in this example interchangably.

	state_space = env.observation_space.n
	action_space = env.action_space.n

	agent = StateToActionLogits(state_space, action_space)
	critic = StateToValue(state_space)

	mean_reward_before_train = evaluate(env, agent, 100)
	print("before training")
	print(f'{mean_reward_before_train=}')

	train(optuna_trial, env, agent, critic, total_timesteps=400_000)

	mean_reward_after_train = evaluate(env, agent, 100)
	print("after training")
	print(f'{mean_reward_after_train=}')

	return mean_reward_after_train


def main():

    sqlite_file = 'optuna_trials.db'
    storage = f'sqlite:///{sqlite_file}'
    optuna_study = optuna.create_study(
        storage=storage,
        study_name=f'{gym_environment} PPO - v1',
        direction="maximize",
        load_if_exists=True,
    )

    optuna_study.optimize(optuna_objective, n_trials=1)


if __name__ == "__main__":
	main()
