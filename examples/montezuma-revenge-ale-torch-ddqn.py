import ast
import copy
import functools
import itertools
from pathlib import Path
import random
from typing import Dict, Tuple
import ale_py
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
# from gymnasium.utils.save_video import save_video
import moviepy
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange
import optuna

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint.tensorboard_utils as tp_tb_utils
import turingpoint.torch_utils as tp_torch_utils
import turingpoint as tp


gym_environment = "ALE/MontezumaRevenge-v5"


class StateToQValues(nn.Module):
    """Copied mostly from Berkeley CS 285/homework_fall2023
    """
    def __init__(self, out_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cnn_layers = []
        in_channels = 4 # for 4 frames (history)
        out_size = np.array((84, 84))
        for kernel_size, stride, out_channels in zip([8, 4, 3], [4, 2, 1], [32, 64, 64]):
            cnn_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
            cnn_layers.append(nn.ReLU())
            in_channels = out_channels
            out_size = (out_size - kernel_size) // stride + 1
        assert out_channels * out_size.prod() == 3136, f'{out_channels=}, {out_size=}'
        self.net = nn.Sequential(
            *cnn_layers,
            nn.Flatten(),
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, out_features),
        )
        self.out_features = out_features # keep for the range of actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> q-values (regression)"""
        assert obs.ndim == 4, f'{obs.shape=}'
        # nn.Flatten() from above skips first dim, so if no batch, we fail to flatten all needed.

        return self.net(obs)


def get_action(parcel: Dict, *, agent: StateToQValues, explore=False):
    """'participant' representing the agent. Epsion greedy strategy:
    1 - epsion - take the "best action" - explotation
    epsilon - take a random action - exploration (can still by chance take the "best action")
    """

    epsilon = parcel.get('epsilon', None)
    assert (not explore) or (epsilon is not None)

    if explore and np.random.binomial(1, epsilon) > 0:
        parcel['action'] = np.random.choice(range(agent.out_features))
    else:
        obs = parcel['obs']
        q_values = agent(torch.tensor(obs).unsqueeze(dim=0))
        _, action = q_values.squeeze().max(dim=0) # or just argmax..
        parcel['action'] = action.item()


def evaluate(env, agent, num_episodes: int) -> float:
    """Collect episodes and calculate the mean total reward."""

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

    for _ in trange(num_episodes, desc="evaluate"):
        _ = evaluate_assembly.launch()
        # Note that we don't clear the rewards in 'rewards_collector', and so we continue to collect.

    total_reward = sum(x['reward'] for x in rewards_collector.get_entries())
    count = sum(1 for _ in rewards_collector.get_entries())

    rewards = [x['reward'] for x in rewards_collector.get_entries()]

    print(f'{np.mean(rewards)=}')
    print(f'{np.max(rewards)=}')
    print(f'{np.min(rewards)=}')

    print(f'{count=}')
    print(f'{total_reward=}')
    print(f'{num_episodes=}')

    return total_reward / num_episodes



def train(optuna_trial, env, actor: StateToQValues, total_timesteps):
    """Given a model (agent) and a critic. Train the model (and the critic) for 'total_timesteps' steps."""

    rendering_env = make_env(render_mode="rgb_array_list", max_episode_steps=1_000)

    actor.eval() # when we'll actually train, we'll say it explicitly below (in learn)

    critic = copy.deepcopy(actor)

    critic.eval()

    use_batch_normilization = optuna_trial.params['use_batch_normilization']

    if use_batch_normilization:
        actor_batch_norm_stats = tp_torch_utils.get_parameters_by_name(actor, ["running_"])
        critic_batch_norm_stats = tp_torch_utils.get_parameters_by_name(critic, ["running_"])

    discount = optuna_trial.suggest_float("discount", 0.99, 0.99) # AKA: gamma
    gradient_steps = optuna_trial.suggest_categorical("gradient_steps", [1])
    batch_size = optuna_trial.suggest_categorical("batch_size", [256])
    learning_starts = optuna_trial.suggest_int("learning_starts", 300, 300) # ? TODO:
    replay_buffer_size = optuna_trial.suggest_categorical("replay_buffer_size", [1_000_000])
    policy_delay = optuna_trial.suggest_categorical("policy_delay", [1])

    replay_buffer_collector = tp_utils.ReplayBufferCollector(
        collect=['obs', 'action', 'reward', 'terminated', 'next_obs'], max_entries=replay_buffer_size)

    per_episode_rewards_collector = tp_utils.Collector(['reward'])

    lr_agent = optuna_trial.suggest_float("lr_agent", 1e-4, 1e-4)

    optimizer_agent = torch.optim.Adam(actor.parameters(), lr=lr_agent) # , weight_decay=5e-6)
    # scheduler_agent = ExponentialLR(optimizer_agent, gamma=0.99)

    max_epsilon = 0.45
    min_epsilon = 0.05

    def set_epsilon(parcel: dict):
        parcel['epsilon'] = (
            min_epsilon
            + (max_epsilon - min_epsilon) * ((total_timesteps - parcel['step']) / total_timesteps)
        )

    videos_path = Path("videos")
    videos_path.mkdir(exist_ok=True)

    def update_critic(agent, critic):
        tau = optuna_trial.suggest_float("tau", 0.01, 0.01) # if needed, can use two different values
        tp_torch_utils.polyak_update(agent.parameters(), critic.parameters(), tau) # agent -> critic
        if use_batch_normilization:
            # Copy running stats, see GH issue #996 (took it from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py)
            tp_torch_utils.polyak_update(actor_batch_norm_stats, critic_batch_norm_stats, 1.0)

    def learn(parcel: dict):

        with tp_torch_utils.start_train(actor), tp_torch_utils.start_train(critic):

            if parcel['step'] == 0:
                parcel['lr_agent'] = optimizer_agent.param_groups[0]['lr']

            replay_buffer = replay_buffer_collector.replay_buffer

            rewards = [x['reward'] for x in replay_buffer[-1000:]]
            parcel['Mean Rewards/train'] = np.mean(rewards) # taking from the replay_buffer ? TODO: !!!

            if parcel['step'] < learning_starts: # we'll start really learning only after we collect some steps
                return

            if parcel['step'] % policy_delay == 0:
                update_critic(actor, critic)

            losses_agent = []

            replay_buffer_dataloader = torch.utils.data.DataLoader(
                replay_buffer,
                batch_size=batch_size,
                sampler=torch.utils.data.RandomSampler(
                    data_source=replay_buffer,
                    replacement=True, # to make stuff faster
                    num_samples=gradient_steps * batch_size
                )
            ) # replacement=True to make it faster

            for batch in replay_buffer_dataloader:

                if len(batch['obs']) < 2:
                    continue

                # Now learn from the batch

                obs = batch['obs'].to(torch.float32)
                action = batch['action']
                reward = batch['reward'].to(torch.float32)
                next_obs = batch['next_obs'].to(torch.float32)
                terminated = batch['terminated']

                # calculate the target

                with torch.no_grad():
                    next_obs_q_values = actor(next_obs)
                    _, next_obs_q_value_idx = next_obs_q_values.max(dim=1)
                    next_obs_q_value = torch.gather(
                        critic(next_obs),
                        dim=1,
                        index=next_obs_q_value_idx.view(-1, 1)
                    ).squeeze()
                    target = reward + torch.where(terminated, 0, discount * next_obs_q_value)

                # optimize agent

                optimizer_agent.zero_grad()

                q_value = torch.gather(actor(obs), dim=1, index=action.view(-1, 1)).squeeze()

                loss = F.mse_loss(q_value, target)
                loss.backward()

                losses_agent.append(loss.item())

                optimizer_agent.step()

                # if parcel['step'] % 40_000 == 0:
                #     scheduler_critic.step()
                #     scheduler_agent.step()
                #     parcel['lr_critic'] = optimizer_critic.param_groups[0]['lr'] # scheduler.get_last_lr()
                #     parcel['lr_agent'] = optimizer_agent.param_groups[0]['lr'] # scheduler.get_last_lr()

            parcel['Loss(agent)/train'] = np.mean(losses_agent)

    def advance(parcel: dict):
        parcel['obs'] = parcel.pop('next_obs')

    def reset_if_needed(parcel: dict):
        terminated = parcel.pop('terminated', False)
        truncated = parcel.pop('truncated', False) 
        if terminated or truncated:
            # some extra logging stuff
            started_step = parcel.get('started_step', 0)
            episode_rewards = [x['reward'] for x in per_episode_rewards_collector.get_entries()]
            per_episode_rewards_collector.clear_entries()
            assert parcel['step'] - started_step == len(episode_rewards), (
                f'{parcel["step"]} - {started_step} != {len(episode_rewards)}'
            )
            parcel['episode_length'] = len(episode_rewards)
            parcel['episode_reward'] = functools.reduce(
                lambda episode_reward, reward: reward + episode_reward * discount,
                reversed(episode_rewards),
                0.0
            )
            parcel['started_step'] = parcel['step']
            # here is what we've actually came to do
            tp_gym_utils.call_reset(parcel, env=env)

    def record_video(parcel: dict, videos_path: Path):
        """When called, either returns immediatly, or renders a one episode video."""

        # Note, we make here usage of "rendering_env" which is slower than "env" as it includes rendering.
        # Those environments are assumed to be similar in all other aspects.

        def get_one_episode_participants():
            yield functools.partial(tp_gym_utils.call_reset, env=rendering_env)
            yield from itertools.cycle([
                functools.partial(get_action, agent=actor),
                functools.partial(tp_gym_utils.call_step, env=rendering_env),
                tp_gym_utils.check_done,
            ])

        one_episode_assembly = tp.Assembly(get_one_episode_participants)
        one_episode_assembly.launch()

        frames = rendering_env.render()

        # Create a video from the frames
        clip = moviepy.ImageSequenceClip([np.uint8(frame) for frame in frames], fps=rendering_env.metadata["render_fps"])

        # Add text
        text = moviepy.TextClip(text=f'After step {parcel["step"]}', font="Lato-Medium.ttf", font_size=14, color='white')
        text = text.with_duration(clip.duration).with_position(("left", "top"))

        # Combine text with the video frames
        final_clip = moviepy.CompositeVideoClip([clip, text])

        # Save the output video
        final_clip.write_videofile(
            videos_path / f"{gym_environment.split('/')[-1]}-DDQN-end-of-step-{parcel['step']}.mp4",
            codec="libx264",
            logger=None
        )

        # save_video(
        #     frames=rendering_env.render(),
        #     video_folder="videos",
        #     episode_index=parcel['step'],
        #     fps=rendering_env.metadata["render_fps"],
        # )

    def get_train_participants():
        with (tp_tb_utils.Logging(
            path=f"runs/{gym_environment}_ddqn_{optuna_trial.datetime_start.strftime('%Y_%B_%d__%H_%M%p')}_study_{optuna_trial.study.study_name}_trial_no_{optuna_trial.number}",
            track=[
                'Mean Rewards/train',
                'Loss(agent)/train',
                'lr_agent',
                'episode_length',
                'episode_reward',
                # 'action',
                'epsilon',

            ]) as logging,
            tp_utils.StepsTracker(total_timesteps=total_timesteps, desc="training steps") as steps_tracker):

            yield functools.partial(tp_gym_utils.call_reset, env=env)
            yield steps_tracker # initialization to 0
            yield from itertools.cycle([
                set_epsilon,
                functools.partial(get_action, agent=actor, explore=True),
                functools.partial(tp_gym_utils.call_step, env=env, save_obs_as="next_obs"),
                replay_buffer_collector,
                learn,
                per_episode_rewards_collector,
                logging,
                functools.partial(
                    tp_utils.call_after_every,
                    every_x_steps=1_000,
                    protected=functools.partial(record_video, videos_path=videos_path)
                ),
                steps_tracker, # can raise Done
                advance,
                reset_if_needed
            ])

    train_assembly = tp.Assembly(get_train_participants)
    
    train_assembly.launch()


def create_network(state_space, action_space, env, use_batch_normilization) -> StateToQValues:
    act = StateToQValues(action_space)
    return act


def create_network_with_optuna_trial(optuna_trial, state_space, action_space, env):
    use_batch_normilization = optuna_trial.suggest_categorical("use_batch_normilization", [False]) # True

    return create_network(state_space, action_space, env, use_batch_normilization)


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

    act = create_network_with_optuna_trial(
        optuna_trial=optuna_trial,
        state_space=state_space,
        action_space=action_space,
        env=env
    )

    name_of_model_file_act = 'act_state.pth'

    if False:
        act.load_state_dict(torch.load(name_of_model_file_act, weights_only=True))

    episodes_for_evaluation = 10

    mean_reward_before_train = evaluate(env, act, episodes_for_evaluation)
    print("before training")
    print(f'{mean_reward_before_train=}')

    total_timesteps = optuna_trial.suggest_categorical("total_timesteps", [10_000]) # 1_000_000
    train(optuna_trial, env, act, total_timesteps=total_timesteps)

    mean_reward_after_train = evaluate(env, act, episodes_for_evaluation)
    print("after training")
    print(f'{mean_reward_after_train=}')

    if True:
        torch.save(act.state_dict(), name_of_model_file_act)

    return mean_reward_after_train


def main():

    # # https://github.com/pytorch/pytorch/issues/51539#issuecomment-1890535975
    # torch.set_flush_denormal(True)

    sqlite_file = 'optuna_trials.db'
    storage = f'sqlite:///{sqlite_file}'
    optuna_study = optuna.create_study(
        storage=storage,
        study_name=f'{gym_environment} DDQN - v1',
        direction="maximize",
        load_if_exists=True,
    )

    optuna_study.optimize(optuna_objective, n_trials=1)


if __name__ == "__main__":
    main()
