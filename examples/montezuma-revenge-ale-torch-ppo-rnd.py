import datetime
import functools
import itertools
from itertools import islice
from pathlib import Path
import random
from typing import Dict, List, Tuple
import numpy as np
import optuna
import ale_py
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import moviepy
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


gym_environment = "ALE/MontezumaRevenge-v5"


class CNNLayers(nn.Module):
    """CNN layers after each there is a non-linearity (ReLU), and also a flattening at the end."""
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cnn_layers = []
        in_channels = in_channels # for 4 frames (history), 1 for a single frame
        out_size = np.array((84, 84))
        for kernel_size, stride, out_channels in zip([8, 4, 3], [4, 2, 1], [32, 64, 64]):
            cnn_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
            cnn_layers.append(nn.ReLU())
            in_channels = out_channels
            out_size = (out_size - kernel_size) // stride + 1
        self.num_ele = out_channels * out_size.prod()
        assert self.num_ele == 3136, f'{out_channels=}, {out_size=}'
        self.net = nn.Sequential(
            *cnn_layers,
            nn.Flatten(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> value (a regression)"""

        return self.net(obs)
    
    @property
    def num_elements(self) -> int:
        return self.num_ele


class StateToActionLogits(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_layers = CNNLayers(in_channels=4)
        self.net = nn.Sequential(
            self.cnn_layers,
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, out_features),
        )
        self.out_features = out_features # keep for the range of actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> actions (logits)"""

        return self.net(obs)


class StateToValue(nn.Module):
    def __init__(self, in_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_layers = CNNLayers(in_channels=4)
        self.net = nn.Sequential(
            self.cnn_layers,
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> value (a regression)"""

        return self.net(obs)


class FixedStateToRandom(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_layers = CNNLayers(in_channels=1)
        latent_dim = 128
        std = np.sqrt(2)
        bias_const = 0.0
        orthogonal = nn.Linear(self.cnn_layers.num_ele, latent_dim)
        torch.nn.init.orthogonal_(orthogonal.weight, std)
        torch.nn.init.constant_(orthogonal.bias, bias_const)
        self.net = nn.Sequential(
            # nn.BatchNorm2d(num_features=1),
            self.cnn_layers,
            nn.Flatten(),
            orthogonal,
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> random (regression 128 values)"""
        # assert obs.ndim == 4, f'{obs.shape=}'
        # nn.Flatten() from above skips first dim, so if no batch, we fail to flatten all needed.

        return self.net(obs)


class StateToRND(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_layer = CNNLayers(in_channels=1)
        latent_dim = 128
        std = np.sqrt(2)
        bias_const = 0.0
        orthogonal = nn.Linear(self.cnn_layer.num_ele, latent_dim)
        torch.nn.init.orthogonal_(orthogonal.weight, std)
        torch.nn.init.constant_(orthogonal.bias, bias_const)
        self.net = nn.Sequential(
            # nn.BatchNorm2d(num_features=1),
            self.cnn_layer,
            nn.Flatten(),
            orthogonal,
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> random (regression 128 values)"""
        # assert obs.ndim == 4, f'{obs.shape=}'
        # nn.Flatten() from above skips first dim, so if no batch, we fail to flatten all needed.

        return self.net(obs)


def get_action(parcel: Dict, *, agent: StateToActionLogits):
    """Picks a random action based on the probabilities that the agent assigns.
    Just needs to account for the fact the the agent actually returns logits rather than probabilities.
    """
    obs = parcel['obs']
    with torch.no_grad():
        logits = agent(torch.tensor(obs).unsqueeze(0))
        action_dist = dist.Categorical(logits=logits)
        if random.random() < parcel.get('epsilon', 0.0):
            action = torch.tensor(random.randrange(agent.out_features))
        action = action_dist.sample()
        parcel['action'] = action.item()
        parcel['log_prob'] = action_dist.log_prob(action) # may be useful for the training (note: still a tensor)


def evaluate(env, agent, num_episodes: int) -> float:

    agent.eval()

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


def collect_episodes(env, agent, num_episodes=40, max_episode_length=160) -> List[List[Dict]]:

    collector = tp_utils.Collector(['obs', 'action', 'log_prob', 'reward', 'next_obs'])

    def advance(parcel: dict):
        parcel['obs'] = parcel.pop('next_obs')

    def set_epsilon(parcel: dict, *, epsilon: float):
        parcel['epsilon'] = epsilon

    def get_episode_participants():
        yield functools.partial(tp_gym_utils.call_reset, env=env)
        yield functools.partial(set_epsilon, epsilon=0.0) # TODO?
        yield from itertools.chain.from_iterable(itertools.repeat(tuple([
                functools.partial(get_action, agent=agent),
                functools.partial(tp_gym_utils.call_step, env=env, save_obs_as='next_obs'),
                collector,
                tp_gym_utils.check_done,
                advance,
        ]), times=max_episode_length))

    episodes_assembly = tp.Assembly(get_episode_participants)

    episodes = [] # it will be a list of lists of dictionaries
    for _ in range(num_episodes):
        _ = episodes_assembly.launch()
        episode = list(collector.get_entries())
        collector.clear_entries()
        episodes.append(episode)
    return episodes


def calc_intrinsic_reward(next_obs, fixed_random, rnd):
    with torch.no_grad():
        fixed_random_value = fixed_random(next_obs).squeeze()
        rnd_values = rnd(next_obs).squeeze()
        assert fixed_random_value.shape == rnd_values.shape, f'{fixed_random_value.shape=}, {rnd_values.shape=}'
        return F.mse_loss(rnd_values, fixed_random_value, reduction="none").mean(dim=-1)


videos_path = Path("videos")
videos_path.mkdir(exist_ok=True)


def train(optuna_trial, env, agent, critic, fixed_random, rnd, critic_int, total_timesteps):

    agent.train()
    critic.train()
    rnd.train()
    critic_int.train()

    rendering_env = make_env(render_mode="rgb_array_list", max_episode_steps=1_000)

    def record_video(num_episodes: int, videos_path: Path):
        """Renders a one episode video."""

        # Note, we make here usage of "rendering_env" which is slower than "env" as it includes rendering.
        # Those environments are assumed to be similar in all other aspects.

        def get_one_episode_participants():
            yield functools.partial(tp_gym_utils.call_reset, env=rendering_env)
            yield from itertools.cycle([
                # FrameStack(),
                functools.partial(get_action, agent=agent),
                functools.partial(tp_gym_utils.call_step, env=rendering_env),
                tp_gym_utils.check_done,
            ])

        one_episode_assembly = tp.Assembly(get_one_episode_participants)
        one_episode_assembly.launch()

        frames = rendering_env.render()

        # Create a video from the frames
        clip = moviepy.ImageSequenceClip([np.uint8(frame) for frame in frames], fps=rendering_env.metadata["render_fps"])

        # Add text
        text = moviepy.TextClip(text=f'After {num_episodes} episodes', font="Lato-Medium.ttf", font_size=14, color='white')
        text = text.with_duration(clip.duration).with_position(("left", "bottom"))

        # Combine text with the video frames
        final_clip = moviepy.CompositeVideoClip([clip, text])

        # Save the output video
        final_clip.write_videofile(
            videos_path / f"{gym_environment.split('/')[-1]}-PPO-RND-after-{num_episodes}-episodes.mp4",
            codec="libx264",
            logger=None
        )

        # save_video(
        #     frames=rendering_env.render(),
        #     video_folder="videos",
        #     episode_index=parcel['step'],
        #     fps=rendering_env.metadata["render_fps"],
        # )

    causality_to_be_accounted_for = True
    normilize_the_rewards = True
    discount = optuna_trial.suggest_float('discount', 0.95, 0.95) # gamma
    gae = optuna_trial.suggest_float('gae_lambda', 0.999, 0.999) # lambda
    gae_int = optuna_trial.suggest_float('gae_lambda_int', 0.99, 0.99) # lambda for intrinsic rewards
    clip_coef = optuna_trial.suggest_float('clip_coef', 0.1, 0.1)
    actor_lr = optuna_trial.suggest_float('actor_lr', 1e-4, 1e-4)
    critic_lr = optuna_trial.suggest_float('critic_lr', 1e-4, 1e-4)

    optimizer = torch.optim.Adam(agent.parameters(), lr=actor_lr)

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    lr_rnd = optuna_trial.suggest_float("lr_rnd", 1e-4, 1e-4)

    optimizer_rnd = torch.optim.Adam(rnd.parameters(), lr=lr_rnd)

    critic_int_optimizer = torch.optim.Adam(critic_int.parameters(), lr=critic_lr) # using same lr as for critic for now

    writer = SummaryWriter(
        f"runs/{gym_environment}_ppo_rnd_{optuna_trial.datetime_start.strftime('%Y_%B_%d__%H_%M%p')}_study_{optuna_trial.study.study_name}_trial_no_{optuna_trial.number}"
    ) # TensorBoard

    num_episodes = 0

    timesteps = 0
    with tqdm(total=total_timesteps, desc="training steps") as pbar:
        while timesteps < total_timesteps:

            episodes = collect_episodes(env, agent, num_episodes=8)
            num_episodes += len(episodes)

            # Now learn from above episodes

            # disclaimer: a lot more of vectorization can be done here.
            # potentially a usage of pandas/polars, numpy, or torch. 
            # I'm KISSing it here.

            obs_batch = []
            action_batch = []
            log_probs_batch = []
            next_obs_batch = []
            values_ext_batch = []
            advantage_ext_batch = []
            values_int_batch = []
            advantage_int_batch = []

            total_rewards = []

            for episode in episodes:
                obs, actions, rewards, log_probs, next_obs = (
                    zip(*((e['obs'], e['action'], e['reward'], e['log_prob'], e['next_obs']) for e in episode))
                )
                obs_batch.extend(obs)
                action_batch.extend(actions)
                log_probs_batch.extend(log_probs)
                next_obs_batch.extend(next_obs)

                values_ext = critic(torch.tensor(np.array(obs))).squeeze(-1).cpu().tolist()
                values_ext.append(0.)
                values_ext_batch.extend(r + discount * v for r, v in zip(rewards, values_ext[1:]))
                advantages_ext = tp_utils.compute_gae(rewards, values_ext, gamma=discount, lambda_=gae)
                advantage_ext_batch.extend(advantages_ext)

                total_reward = sum(rewards) # TODO: discounted reward?
                total_rewards.append(total_reward)

                values_int: List = critic_int(torch.tensor(np.array(next_obs))).squeeze(-1).cpu().tolist()
                values_int.append(0.)
                rewards_int = calc_intrinsic_reward(torch.tensor(np.array(next_obs))[:, -1:, :, :], fixed_random, rnd)
                values_int_batch.extend(r + discount * v for r, v in zip(rewards_int, values_int[1:]))
                advantages_int = tp_utils.compute_gae(rewards_int, values_int, gamma=discount, lambda_=gae_int)
                advantage_int_batch.extend(advantages_int.tolist())

            obs_tensor = torch.tensor(np.array(obs_batch))
            actions_tensor = torch.tensor(action_batch)
            advantages_ext_tensor = torch.tensor(advantage_ext_batch)
            log_probs_batch_tensor = torch.tensor(log_probs_batch)
            next_obs_batch_tensor = torch.tensor(np.array(next_obs_batch))
            values_ext_tensor = torch.tensor(values_ext_batch, dtype=torch.float32)
            values_int_tensor = torch.tensor(values_int_batch, dtype=torch.float32)
            advantages_int_tensor = torch.tensor(advantage_int_batch)

            assert len(log_probs_batch_tensor) == len(advantages_ext_tensor), f'{len(log_probs_batch_tensor)=}, {len(advantages_ext_tensor)=}'

            def normalize(x: torch.Tensor) -> torch.Tensor:
                x_mean = x.mean()
                return (x - x_mean) / (x.std() + 1e-5)

            if normilize_the_rewards:
                advantages_ext_tensor = normalize(advantages_ext_tensor)
                advantages_int_tensor = normalize(advantages_int_tensor)
                values_ext_tensor = normalize(values_ext_tensor)
                values_int_tensor = normalize(values_int_tensor)
                # TODO: all always?

            timesteps += len(log_probs_batch_tensor)

            ds = TensorDataset(obs_tensor, advantages_ext_tensor, values_ext_tensor, actions_tensor, log_probs_batch_tensor, next_obs_batch_tensor, values_int_tensor, advantages_int_tensor)
            dl = DataLoader(ds, batch_size=128, shuffle=True)

            losses_rnd = []
            for epoch in range(4):
                for o, adv_ext, v_ext, act, l_p, next_obs, v_int, adv_int in islice(dl, None):

                    fixed_random_value = fixed_random(next_obs[:, -1:, :, :])
                    rnd_values = rnd(next_obs[:, -1:, :, :])

                    # optimize rnd

                    optimizer_rnd.zero_grad()

                    loss_rnd = F.mse_loss(rnd_values, fixed_random_value)
                    loss_rnd.backward()

                    losses_rnd.append(loss_rnd.item())

                    optimizer_rnd.step()

                    #

                    logits = agent(o)
                    action_dist = dist.Categorical(logits=logits)
                    log_probs = action_dist.log_prob(act)

                    # assert probs.shape == probs_batch_tensor.shape, f'{probs.shape=}, {probs_batch_tensor.shape=}'

                    optimizer.zero_grad()

                    ratio = (log_probs - l_p).exp()
                    loss1 = ratio * adv_ext
                    loss2 = torch.clip(ratio, 1. - clip_coef, 1. + clip_coef) * adv_ext

                    coef_int = 2.0
                    coef_ext = 1.0

                    loss = -(coef_ext * torch.min(loss1, loss2).mean() + coef_int * adv_int.mean())# we want to maximize

                    loss.backward()

                    optimizer.step()

                    # optimize the critic

                    critic_optimizer.zero_grad()

                    critic_loss = F.mse_loss(critic(o).squeeze(-1), v_ext)

                    critic_loss.backward()

                    critic_optimizer.step()

                    # optimize the intrinsic critic

                    critic_int_optimizer.zero_grad()

                    critic_int_loss = F.mse_loss(critic_int(next_obs).squeeze(-1), v_int)

                    critic_int_loss.backward()

                    critic_int_optimizer.step()

                    #

                    writer.add_scalar("Loss/train", loss, timesteps)
                    writer.add_scalar("Loss Critic/train", critic_loss, timesteps)
                    writer.add_scalar("Loss RND/train", loss_rnd, timesteps)

            writer.add_scalar("Mean Rewards/train", np.mean(total_rewards), timesteps)

            pbar.update(len(advantages_ext_tensor))

            if num_episodes % 20 == 0:
                record_video(num_episodes, videos_path)

    writer.flush()


def make_env(**kwargs) -> gym.Env:
    env = gym.make(gym_environment, frameskip=1, **kwargs) # (210, 160, 3)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=True,
    )
    env = FrameStackObservation(env, stack_size=4)

    return env


def create_network(state_space, action_space, env, use_batch_normilization) -> Tuple[
        StateToActionLogits,
        StateToValue,
        FixedStateToRandom,
        StateToRND,
        StateToValue,
    ]:
    act = StateToActionLogits(state_space, action_space)
    critic = StateToValue(state_space)
    fixed_random = FixedStateToRandom()
    for param in fixed_random.parameters():
        param.requires_grad = False
    rnd = StateToRND()
    critic_int = StateToValue(state_space)
    return act, critic, fixed_random, rnd, critic_int


def create_network_with_optuna_trial(optuna_trial, state_space, action_space, env):
    use_batch_normilization = optuna_trial.suggest_categorical("use_batch_normilization", [False]) # True

    return create_network(state_space, action_space, env, use_batch_normilization)


def optuna_objective(optuna_trial):

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    gym.register_envs(ale_py)

    env = make_env()

    env.reset(seed=1)

    # state and obs/observations are used in this example interchangably.

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent, critic, fixed_random, rnd, critic_int = create_network_with_optuna_trial(optuna_trial, state_space, action_space, env)

    episodes_for_evaluation = 10

    mean_reward_before_train = evaluate(env, agent, episodes_for_evaluation)
    print("before training")
    print(f'{mean_reward_before_train=}')

    train(optuna_trial, env, agent, critic, fixed_random, rnd, critic_int, total_timesteps=400_000) # 4_000_000)

    mean_reward_after_train = evaluate(env, agent, episodes_for_evaluation)
    print("after training")
    print(f'{mean_reward_after_train=}')

    return mean_reward_after_train


def main():

    sqlite_file = 'optuna_trials.db'
    storage = f'sqlite:///{sqlite_file}'
    optuna_study = optuna.create_study(
        storage=storage,
        study_name=f'{gym_environment} PPO - RND - v1',
        direction="maximize",
        load_if_exists=True,
    )

    optuna_study.optimize(optuna_objective, n_trials=1)


if __name__ == "__main__":
    main()

