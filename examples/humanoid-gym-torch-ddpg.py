import ast
import copy
import datetime
import functools
import itertools
import random
from typing import Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium.utils.save_video import save_video
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange
from contextlib import contextmanager
import optuna

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint.tensorboard_utils as tp_tb_utils
import turingpoint.torch_utils as tp_torch_utils
import turingpoint as tp


gym_environment = 'Humanoid-v5'


class StateToAction(nn.Module):
    def __init__(self, in_features, out_actions, env, *args, **kwargs):
        use_batch_normilization = kwargs.pop('use_batch_normilization', None)
        super().__init__(*args, **kwargs)
        layers = []
        in_f = in_features
        for out_f in [400, 300]:
            if use_batch_normilization:
                layers.append(nn.BatchNorm1d(in_f))
            layers.append(nn.Linear(in_features=in_f, out_features=out_f))
            layers.append(nn.ReLU())
            in_f = out_f
        layers.append(nn.Linear(in_features=in_f, out_features=out_actions))
        self.net = nn.Sequential(*layers)

        # # action rescaling (copied from https://docs.cleanrl.dev/rl-algorithms/ddpg/#implementation-details)
        # self.register_buffer(
        #     "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        # )
        # self.register_buffer(
        #     "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        # )


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> action (regression)"""

        return F.tanh(self.net(obs)) * 4.0
        # x = self.net(obs)
        # return x * self.action_scale + self.action_bias # (copied from https://docs.cleanrl.dev/rl-algorithms/ddpg/#implementation-details)


class StateActionToQValue(nn.Module):
    def __init__(self, in_features, in_actions, *args, **kwargs):
        use_batch_normilization = kwargs.pop('use_batch_normilization', None)
        layers_critic = kwargs.pop('layers_critic', None)
        super().__init__(*args, **kwargs)
        layers = []
        in_f = in_features + in_actions
        for out_f in layers_critic:
            if use_batch_normilization:
                layers.append(nn.BatchNorm1d(in_f))
            layers.append(nn.Linear(in_features=in_f, out_features=out_f))
            layers.append(nn.ReLU())
            in_f = out_f
        self.state_net = nn.Sequential(*layers)
        layers.append(nn.Linear(in_features=in_f, out_features=1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """obs, action -> q-values (regression)"""

        return self.net(torch.concat((obs, actions), dim=1))


def get_action(parcel: Dict, *, agent: StateToAction, explore=False):
    """'participant' representing the agent. when 'explore' adds noise. 
    """

    obs = parcel['obs']
    assert not agent.training # the BN above needs more than 1 sample during training..
    action = agent(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
    if explore:
        noise = torch.randn_like(action) * 0.1
        action = torch.clamp(action + noise, -0.4, 0.4)
    parcel['action'] = action.squeeze(0).detach().numpy() # .item()


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



@contextmanager
def start_train(model: nn.Module):
    try:
        model.train()
        yield model
    finally:
        model.eval()


def train(optuna_trial, env, actor: StateToAction, critic: StateActionToQValue, total_timesteps):
    """Given a model (agent) and a critic. Train the model (and the critic) for 'total_timesteps' steps."""

    rendering_env = gym.make(gym_environment, render_mode="rgb_array_list")

    actor.eval() # when we'll actually train, we'll say it explicitly below (in learn)
    critic.eval() # same here

    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)

    target_actor.eval()
    target_critic.eval()

    use_batch_normilization = optuna_trial.params['use_batch_normilization']

    if use_batch_normilization:
        actor_batch_norm_stats = tp_torch_utils.get_parameters_by_name(actor, ["running_"])
        critic_batch_norm_stats = tp_torch_utils.get_parameters_by_name(critic, ["running_"])
        actor_batch_norm_stats_target = tp_torch_utils.get_parameters_by_name(target_actor, ["running_"])
        critic_batch_norm_stats_target = tp_torch_utils.get_parameters_by_name(target_critic, ["running_"])

    discount = optuna_trial.suggest_categorical("discount", [0.995]) # AKA: gamma
    gradient_steps = optuna_trial.suggest_categorical("gradient_steps", [2])
    batch_size = optuna_trial.suggest_categorical("batch_size", [256])
    learning_starts = optuna_trial.suggest_categorical("learning_starts", [25_000])
    replay_buffer_size = optuna_trial.suggest_categorical("replay_buffer_size", [1_000_000])
    policy_delay = optuna_trial.suggest_categorical("policy_delay", [2])

    replay_buffer_collector = tp_utils.ReplayBufferCollector(
        collect=['obs', 'action', 'reward', 'terminated', 'truncated', 'next_obs'], max_entries=replay_buffer_size)

    per_episode_rewards_collector = tp_utils.Collector(['reward'])

    lr_agent = optuna_trial.suggest_categorical("lr_agent", [1e-4])
    lr_critic = optuna_trial.suggest_categorical("lr_critic", [1e-3])

    optimizer_agent = torch.optim.Adam(actor.parameters(), lr=lr_agent)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    # scheduler_agent = ExponentialLR(optimizer_agent, gamma=0.99)
    # scheduler_critic = ExponentialLR(optimizer_critic, gamma=0.99)

    # max_epsilon = 0.15
    # min_epsilon = 0.05

    def update_target(target_agent, target_critic, agent, critic):
        tau = optuna_trial.suggest_categorical("tau", [0.001]) # if needed, can use two different values
        tp_torch_utils.polyak_update(agent.parameters(), target_agent.parameters(), tau) # agent -> target_agent
        tp_torch_utils.polyak_update(critic.parameters(), target_critic.parameters(), tau) # critic -> target_critic
        if use_batch_normilization:
            # Copy running stats, see GH issue #996 (took it from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py)
            tp_torch_utils.polyak_update(critic_batch_norm_stats, critic_batch_norm_stats_target, 1.0)
            tp_torch_utils.polyak_update(actor_batch_norm_stats, actor_batch_norm_stats_target, 1.0)

    def learn(parcel: dict):

        with start_train(actor), start_train(critic):

            if parcel['step'] == 0:
                parcel['lr_agent'] = optimizer_agent.param_groups[0]['lr']
                parcel['lr_critic'] = optimizer_critic.param_groups[0]['lr']

            replay_buffer = replay_buffer_collector.replay_buffer

            rewards = [x['reward'] for x in replay_buffer[-1000:]]
            parcel['Mean Rewards/train'] = np.mean(rewards) # taking from the replay_buffer ? TODO: !!!

            if parcel['step'] < learning_starts: # we'll start really learning only after we collect some steps
                return

            if parcel['step'] % policy_delay == 0:
                update_target(target_actor, target_critic, actor, critic)

            losses_agent = []
            losses_critic = []

            replay_buffer_dataloader = torch.utils.data.DataLoader(
                replay_buffer, batch_size=batch_size, shuffle=True)

            for _, batch in zip(range(gradient_steps), replay_buffer_dataloader):

                if len(batch['obs']) < 2:
                    continue

                # Now learn from the batch

                obs = batch['obs'].to(torch.float32)
                action = batch['action']
                reward = batch['reward'].to(torch.float32)
                next_obs = batch['next_obs'].to(torch.float32)
                terminated = batch['terminated']
                # truncated = batch['truncated']

                with torch.no_grad():
                    next_actions = target_actor(next_obs)
                    next_obs_q_values = target_critic(next_obs, next_actions).squeeze()
                    target = reward + torch.where(terminated, 0, discount * next_obs_q_values)

                optimizer_critic.zero_grad()

                q_value = critic(obs, action).squeeze()

                loss = F.mse_loss(q_value, target)

                loss.backward()

                losses_critic.append(loss.item())

                optimizer_critic.step()

                optimizer_agent.zero_grad()

                loss = -critic(obs, actor(obs)).mean() # let's maximize this value (hence the minus sign)

                loss.backward()

                optimizer_agent.step()

                losses_agent.append(loss.item())

                # if parcel['step'] % 5000 == 0:
                #     scheduler_critic.step()
                #     scheduler_agent.step()
                #     parcel['lr_critic'] = optimizer_critic.param_groups[0]['lr'] # scheduler.get_last_lr()
                #     parcel['lr_agent'] = optimizer_agent.param_groups[0]['lr'] # scheduler.get_last_lr()

            parcel['Loss(agent)/train'] = np.mean(losses_agent)
            # losses_agent_std = np.std(losses_agent)
            # parcel['Loss(agent)+std/train'] = parcel['Loss(agent)/train'] + losses_agent_std
            # parcel['Loss(agent)-std/train'] = parcel['Loss(agent)/train'] - losses_agent_std
            parcel['Loss(critic)/train'] = np.mean(losses_critic)
            # losses_critic_std = np.std(losses_critic)
            # parcel['Loss(critic)+std/train'] = parcel['Loss(critic)/train'] + losses_critic_std
            # parcel['Loss(critic)-std/train'] = parcel['Loss(critic)/train'] - losses_critic_std

    # def set_epsilon(parcel: dict):
    #     parcel['epsilon'] = (
    #         min_epsilon
    #         + (max_epsilon - min_epsilon) * ((total_timesteps - parcel['step']) / total_timesteps)
    #     )

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
            parcel['episode_reward'] = functools.reduce(lambda r, x: x + r * discount, reversed(episode_rewards), 0)
            parcel['started_step'] = parcel['step']
            # here is what we've actually came to do
            tp_gym_utils.call_reset(parcel, env=env)

    def take_interesting_info(parcel: dict):
        parcel.update(parcel.pop('info'))

    def record_video(parcel: dict):
        """When called, either returns immediatly, or renders a one episode video."""

        if parcel['step'] % 5000 != 0:
            return

        # Note, we make here usage of "rendering_env" which is slower than "env" as it includes rendering.
        # Those environments are assumed to be similar in all other aspects.

        def get_one_episode_participants():
            yield functools.partial(tp_gym_utils.call_reset, env=rendering_env)
            yield from itertools.cycle([
                functools.partial(get_action, agent=actor, explore=False),
                functools.partial(tp_gym_utils.call_step, env=rendering_env),
                tp_gym_utils.check_done,
            ])

        one_episode_assembly = tp.Assembly(get_one_episode_participants)
        one_episode_assembly.launch()

        save_video(
            frames=rendering_env.render(),
            video_folder="videos",
            episode_index=parcel['step'],
            fps=rendering_env.metadata["render_fps"],
        )

    def get_train_participants():
        with (tp_tb_utils.Logging(
            path=f"runs/humanoid_ddpg_{datetime.datetime.now().strftime('%Y_%B_%d__%H_%M%p')}_{discount=}",
            track=[
                'Mean Rewards/train',
                'Loss(agent)/train',
                'Loss(critic)/train',
                # 'lr_critic',
                # 'lr_agent',
                'episode_length',
                'episode_reward',
                # 'action',
                # 'noise',

                'reward_survive',
                'reward_forward',
                'reward_ctrl',

                # 'Loss(agent)+std/train',
                # 'Loss(agent)-std/train',
                # 'Loss(critic)+std/train',
                # 'Loss(critic)-std/train',
            ]) as logging,
            tp_utils.StepsTracker(total_timesteps=total_timesteps, desc="training steps") as steps_tracker):

            yield functools.partial(tp_gym_utils.call_reset, env=env)
            yield steps_tracker # initialization to 0
            yield from itertools.cycle([
                # set_epsilon,
                functools.partial(get_action, agent=actor, explore=True),
                functools.partial(tp_gym_utils.call_step, env=env, save_obs_as="next_obs"),
                replay_buffer_collector,
                learn,
                per_episode_rewards_collector,
                take_interesting_info, # those will be also logged.
                logging,
                record_video,
                steps_tracker, # can raise Done
                advance,
                reset_if_needed
            ])

    train_assembly = tp.Assembly(get_train_participants)
    
    train_assembly.launch()


def create_networks(state_space, action_space, env, use_batch_normilization, layers_critic) -> Tuple[StateToAction, StateActionToQValue]:
    act = StateToAction(
        state_space,
        out_actions=action_space,
        env=env,
        use_batch_normilization=use_batch_normilization
    ) # This is the agent
    critic = StateActionToQValue(
        state_space,
        action_space,
        use_batch_normilization=use_batch_normilization,
        layers_critic=layers_critic
    ) # This is the Q value
    return act, critic


def create_networks_with_optuna_trial(optuna_trial, state_space, action_space, env):
    use_batch_normilization = optuna_trial.suggest_categorical("use_batch_normilization", [False])
    layers_critic = optuna_trial.suggest_categorical("layers_critic", ["[300, 400, 300, 400]"])
    layers_critic = ast.literal_eval(layers_critic)

    return create_networks(state_space, action_space, env, use_batch_normilization, layers_critic)


def optuna_objective(optuna_trial, env):

    # state and obs/observations are used in this example interchangably.

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[-1]

    # print(f'{state_space=}')
    # print(f'{action_space=}')

    act, critic = create_networks_with_optuna_trial(
        optuna_trial=optuna_trial,
        state_space=state_space,
        action_space=action_space,
        env=env
    )

    name_of_model_file_act = 'act_state.pth'
    name_of_model_file_critic = 'critic_state.pth'

    if False:
        act.load_state_dict(torch.load(name_of_model_file_act, weights_only=True))
        critic.load_state_dict(torch.load(name_of_model_file_critic, weights_only=True))

    mean_reward_before_train = evaluate(env, act, 100)
    print("before training")
    print(f'{mean_reward_before_train=}')

    total_timesteps = optuna_trial.suggest_categorical("total_timesteps", [40_000])
    train(optuna_trial, env, act, critic, total_timesteps=total_timesteps)

    mean_reward_after_train = evaluate(env, act, 100)
    print("after training")
    print(f'{mean_reward_after_train=}')

    torch.save(act.state_dict(), name_of_model_file_act)
    torch.save(critic.state_dict(), name_of_model_file_critic)

    return mean_reward_after_train


def main():

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    sqlite_file = 'optuna_trials.db'
    storage = f'sqlite:///{sqlite_file}'
    optuna_study = optuna.create_study(
        storage=storage,
        study_name=f'{gym_environment} DDPG',
        direction="maximize",
        load_if_exists=True,
    )

    env = gym.make(gym_environment)
    env.reset(seed=1)

    optuna_study.optimize(functools.partial(optuna_objective, env=env), n_trials=1)


if __name__ == "__main__":
    main()
