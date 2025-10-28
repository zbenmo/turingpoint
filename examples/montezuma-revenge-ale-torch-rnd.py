import datetime
import functools
import itertools
from itertools import groupby, islice
from pathlib import Path
import random
from typing import Dict, Generator, List, Sequence, Tuple
import numpy as np
import optuna
import ale_py
from gymnasium.vector.sync_vector_env import SyncVectorEnv
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import moviepy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as dist

import turingpoint.gymnasium_utils as tp_gym_utils
import turingpoint.utils as tp_utils
import turingpoint as tp
import turingpoint.torch_utils as tp_torch_utils
import turingpoint.tensorboard_utils as tp_tb_utils


from line_profiler import profile


# add device selection and CUDA tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


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
        self.num_ele = (out_channels * out_size.prod()).item()
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
            nn.Linear(512, out_features.item()),
        )
        self.out_features = out_features # keep for the range of actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> actions (logits)"""

        return self.net(obs)


class StateToValue(nn.Module):
    def __init__(self, in_features, in_channels=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_layers = CNNLayers(in_channels=in_channels)
        self.net = nn.Sequential(
            self.cnn_layers,
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Softplus(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> non-negagive value (a regression)"""

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
            orthogonal,
            # nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs -> random (regression 128 values)"""

        return self.net(obs)


class StateToRND_Predictor(nn.Module): # Renamed for clarity
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnn_layer = CNNLayers(in_channels=1) # Same CNN backbone
        latent_dim = 128

        # Predictor Network (deeper, trained)
        self.net = nn.Sequential(
            self.cnn_layer,
            nn.Linear(self.cnn_layer.num_ele, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """Scale observations."""
    return obs.astype(np.float32) / 255.0


def make_env(seed, **kwargs) -> gym.Env:
    env = gym.make(gym_environment, frameskip=1, **kwargs) # (210, 160, 3)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=1,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)

    return env


def get_action(parcel: Dict, *, agent: StateToActionLogits):
    """Picks a random action based on the probabilities that the agent assigns.
    Just needs to account for the fact the the agent actually returns logits rather than probabilities.
    """
    obs = parcel['obs']
    obs = preprocess_observation(obs)
    obs_tensor = torch.tensor(obs)
    obs_tensor = obs_tensor.reshape(-1, *obs_tensor.shape[-3:]) # from batch x num envs x ..rest to (batch x num envs) x ..rest
    with torch.no_grad():
        logits = agent(obs_tensor)
        action_dist = dist.Categorical(logits=logits)
        if random.random() < parcel.get('epsilon', 0.0):
            action = torch.tensor(random.randrange(agent.out_features))
        else:
            action = action_dist.sample()
        parcel['action'] = action.numpy()
        parcel['log_prob'] = action_dist.log_prob(action) # may be useful for the training (note: still a tensor)


def get_single_action(parcel: Dict, *, agent: StateToActionLogits):
    """Picks a random action based on the probabilities that the agent assigns.
    Just needs to account for the fact the the agent actually returns logits rather than probabilities.
    """
    obs = parcel['obs']
    obs = preprocess_observation(obs)
    with torch.no_grad():
        obs_tensor = torch.tensor(obs)
        obs_tensor = obs_tensor.unsqueeze(0)
        logits = agent(obs_tensor)
        action_dist = dist.Categorical(logits=logits)
        # if random.random() < parcel.get('epsilon', 0.0):
        #     action = torch.tensor(random.randrange(agent.out_features))
        # else:
        action = action_dist.sample()
        parcel['action'] = action.item()
        # parcel['log_prob'] = action_dist.log_prob(action) # may be useful for the training (note: still a tensor)


def evaluate(env, agent, num_episodes: int) -> float:

    agent.eval()

    rewards_collector = tp_utils.Collector(['reward'])

    def get_participants():
        yield functools.partial(tp_gym_utils.call_reset, env=env)
        yield from itertools.cycle([
                functools.partial(get_single_action, agent=agent),
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


videos_path = Path("videos")
videos_path.mkdir(exist_ok=True)


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    @property
    def std(self):
        return np.sqrt(self.var)


time_steps = 128 # for each learning round
num_parallel_envs = 32 # vectorized env


def train(optuna_trial, vec_env, actor, critic, fixed_random, rnd, critic_int, total_timesteps):

    actor.train()
    critic.train()
    rnd.train()
    critic_int.train()

    rendering_env = make_env(seed=1, render_mode="rgb_array_list", max_episode_steps=1_000)

    def record_video(parcel: dict, videos_path: Path):
        """Renders a one episode video."""
        num_steps: int = parcel['step'] + 1

        # Note, we make here usage of "rendering_env" which is slower than "env" as it includes rendering.
        # Those environments are assumed to be similar in all other aspects.

        def get_one_episode_participants():
            yield functools.partial(tp_gym_utils.call_reset, env=rendering_env)
            yield from itertools.cycle([
                # FrameStack(),
                functools.partial(get_single_action, agent=actor),
                functools.partial(tp_gym_utils.call_step, env=rendering_env),
                tp_gym_utils.check_done,
            ])

        one_episode_assembly = tp.Assembly(get_one_episode_participants)
        one_episode_assembly.launch()

        frames = rendering_env.render()

        # Create a video from the frames
        clip = moviepy.ImageSequenceClip([np.uint8(frame) for frame in frames], fps=rendering_env.metadata["render_fps"])

        # Add text
        text = moviepy.TextClip(text=f'After {num_steps} steps', font="Lato-Medium.ttf", font_size=14, color='white')
        text = text.with_duration(clip.duration).with_position(("left", "bottom"))

        # Combine text with the video frames
        final_clip = moviepy.CompositeVideoClip([clip, text])

        # Save the output video
        final_clip.write_videofile(
            videos_path / f"{gym_environment.split('/')[-1]}-RND-after-{num_steps}-steps.mp4",
            codec="libx264",
            logger=None
        )

        # save_video(
        #     frames=rendering_env.render(),
        #     video_folder="videos",
        #     episode_index=parcel['step'],
        #     fps=rendering_env.metadata["render_fps"],
        # )

    collector = tp_utils.Collector(['obs', 'action', 'log_prob', 'reward', 'next_obs', 'terminated', 'truncated'])

    def advance(parcel: dict):
        parcel['obs'] = parcel.pop('next_obs')

    normilize_the_rewards = False
    discount = optuna_trial.suggest_float('discount', 0.95, 0.95) # gamma
    gae = optuna_trial.suggest_float('gae_lambda', 0.999, 0.999) # lambda
    gae_int = optuna_trial.suggest_float('gae_lambda_int', 0.99, 0.99) # lambda for intrinsic rewards
    clip_coef = optuna_trial.suggest_float('clip_coef', 0.1, 0.1)
    actor_lr = optuna_trial.suggest_float('actor_lr', 2.5e-4, 2.5e-4)
    critic_lr = optuna_trial.suggest_float('critic_lr', 2.5e-4, 2.5e-4)

    entropy_coef = optuna_trial.suggest_float('entropy_coef', 0.001, 0.1)

    coef_int = optuna_trial.suggest_float('coef_int', 0.01, 0.5)
    coef_ext = optuna_trial.suggest_float('coef_ext', 1.0, 1.0)

    should_record_video = optuna_trial.suggest_categorical('record_video', [False, True])

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    lr_rnd = optuna_trial.suggest_float("lr_rnd", 5e-4, 5e-4)

    optimizer_rnd = torch.optim.Adam(rnd.parameters(), lr=lr_rnd)

    critic_int_optimizer = torch.optim.Adam(critic_int.parameters(), lr=critic_lr) # using same lr as for critic for now

    def calc_intrinsic_reward(next_obs, fixed_random, rnd):
        with torch.no_grad():
            fixed_random_value = fixed_random(next_obs) #.squeeze()
            rnd_values = rnd(next_obs) #.squeeze()
            assert fixed_random_value.shape == rnd_values.shape, f'{fixed_random_value.shape=}, {rnd_values.shape=}'
            return F.mse_loss(rnd_values, fixed_random_value, reduction="none").mean(dim=-1)

    def extract_lists_from_generator_of_dicts(generator_of_dicts: Generator[Dict, None, None], keys: Sequence[str]) -> Sequence[List]:
        return tuple(map(list, zip(*((e[k] for k in keys) for e in generator_of_dicts))))

    intrinsic_rms = RunningMeanStd()

    @tp_utils.track_calls
    @profile
    def learn(parcel: dict):
        obs, action, reward, log_prob, next_obs, terminated, truncated = extract_lists_from_generator_of_dicts(
            generator_of_dicts=collector.get_entries(),
            keys=('obs', 'action', 'reward', 'log_prob', 'next_obs', 'terminated', 'truncated')
        )
        collector.clear_entries()

        assert len(obs) == time_steps, f'{len(obs)=}'

        obs = preprocess_observation(np.array(obs))

        next_obs = np.array(next_obs)
        next_obs = next_obs.reshape(-1, *next_obs.shape[2:])

        orig_next_obs = next_obs[:, -1:, :, :].copy()
        intrinsic_rms.update(orig_next_obs)

        next_obs = preprocess_observation(next_obs)

        # obs_tensor = torch.tensor(obs)
        # obs_tensor = obs_tensor.reshape(-1, *obs_tensor.shape[2:])
        obs_tensor = torch.from_numpy(obs).reshape(-1, *obs.shape[2:]).to(device, non_blocking=True)

        assert obs_tensor.shape == (time_steps * num_parallel_envs, 4, 84, 84), f'{obs_tensor.shape=}'

        action_tensor = torch.from_numpy(np.array(action)).reshape(-1).to(device, non_blocking=True)

        assert action_tensor.shape == (time_steps * num_parallel_envs,), f'{action_tensor.shape=}'

        reward_tensor = torch.from_numpy(np.array(reward, dtype=np.float32)).reshape(-1).to(device, non_blocking=True)

        assert reward_tensor.shape == (time_steps * num_parallel_envs,), f'{reward_tensor.shape=}'

        log_prob_tensor = torch.from_numpy(np.array(log_prob, dtype=np.float32)).reshape(-1).to(device, non_blocking=True)

        assert log_prob_tensor.shape == (time_steps * num_parallel_envs,), f'{log_prob_tensor.shape=}'

        next_obs_tensor = torch.from_numpy(next_obs).to(device, non_blocking=True)
        # next_obs_tensor = next_obs_tensor.reshape(-1, *next_obs_tensor.shape[2:])

        assert next_obs_tensor.shape == (time_steps * num_parallel_envs, 4, 84, 84), f'{next_obs_tensor.shape=}'

        parcel['log_prob_mean'] = log_prob_tensor.mean().item()

        for trunc, term in zip(truncated, terminated):
            trunc[-1] = not term[-1]

        terminated = np.array(terminated).reshape(-1)
        truncated = np.array(truncated).reshape(-1)

        done_flags = np.logical_or(terminated, truncated)
        episode_fragment_index = np.cumsum(done_flags)

        values_ext_batch = []
        advantage_ext_batch = []
        values_int_batch = []
        advantage_int_batch = []

        with torch.no_grad():
            for _, group in groupby(range(len(episode_fragment_index)), key=lambda i: episode_fragment_index[i]):
                indices = list(group)
                last_index = indices[-1]
                relevant_obs = obs_tensor[indices]
                relevant_reward = reward_tensor[indices]
                if terminated[last_index]:
                    # add 0 to the end
                    values_ext = critic(relevant_obs).squeeze(-1).cpu().tolist()
                    values_ext.append(0.)
                else:
                    # add the value estimate of the last "next_obs"
                    relevant_obs = torch.cat([relevant_obs, next_obs_tensor[last_index:last_index+1]], axis=0)
                    values_ext = critic(relevant_obs).squeeze(-1).cpu().tolist()
                relevant_reward = relevant_reward.cpu().tolist()
                values_ext_batch.extend(r + discount * v for r, v in zip(relevant_reward, values_ext[1:]))
                advantages_ext = tp_utils.compute_gae(relevant_reward, values_ext, gamma=discount, lambda_=gae)
                advantage_ext_batch.extend(advantages_ext)

        total_rewards = [] # strange there will be only one reward per learn call TODO:

        total_reward = sum(reward) # TODO: discounted reward?
        total_rewards.append(total_reward)

        values_int_batch = []
        advantage_int_batch = []
        rewards_ints = []

        orig_next_obs_tensor = torch.from_numpy(orig_next_obs).to(device, non_blocking=True)
        mean = torch.tensor(intrinsic_rms.mean, dtype=torch.float32, device=device)
        std = torch.tensor(intrinsic_rms.std, dtype=torch.float32, device=device)
        orig_next_obs_tensor = (orig_next_obs_tensor - mean) / (std + 1e-8)
        orig_next_obs_tensor = orig_next_obs_tensor.clamp(-5.0, 5.0)

        with torch.no_grad():
            for _, group in groupby(range(len(episode_fragment_index)), key=lambda i: episode_fragment_index[i]):
                indices = list(group)
                last_index = indices[-1]
                relevant_next_obs = orig_next_obs_tensor[indices]
                if terminated[last_index]:
                    values_int = critic_int(relevant_next_obs).squeeze(-1).cpu().tolist()
                    values_int.append(0.)
                else:
                    # TODO: the same?
                    values_int = critic_int(relevant_next_obs).squeeze(-1).cpu().tolist()
                    values_int.append(0.)
                rewards_int = calc_intrinsic_reward(relevant_next_obs, fixed_random, rnd)
                rewards_int = rewards_int.numpy()
                rewards_ints.extend(rewards_int.tolist())
                values_int_batch.extend(r + discount * v for r, v in zip(rewards_int, values_int[1:]))
                advantages_int = tp_utils.compute_gae(rewards_int, values_int, gamma=discount, lambda_=gae_int)
                advantage_int_batch.extend(advantages_int.tolist())

        advantages_ext_tensor = torch.tensor(advantage_ext_batch)
        values_ext_tensor = torch.tensor(values_ext_batch, dtype=torch.float32)
        values_int_tensor = torch.tensor(values_int_batch, dtype=torch.float32)
        advantages_int_tensor = torch.tensor(advantage_int_batch)

        parcel['mean_int_reward'] = np.mean(rewards_ints)
        parcel['values_ext'] = values_ext_tensor.mean().item()
        parcel['values_int'] = values_int_tensor.mean().item()
        parcel['advantages_int'] = advantages_int_tensor.mean().item()

        ds = TensorDataset(
            obs_tensor,
            advantages_ext_tensor,
            values_ext_tensor,
            action_tensor,
            log_prob_tensor,
            orig_next_obs_tensor,
            values_int_tensor,
            advantages_int_tensor,
        )
        dl = DataLoader(ds, batch_size=256, shuffle=True)

        parcel['coef_int'] = coef_int
        parcel['coef_ext'] = coef_ext
        parcel['entropy_coef'] = entropy_coef

        losses_rnd = []
        losses_actor = []
        losses_critic = []
        losses_critic_int = []
        entropies = []
        for epoch in range(4):
            for o, adv_ext, v_ext, act, l_p, next_obs, v_int, adv_int in islice(dl, None): # 128*32/256=16

                #

                logits = actor(o)
                action_dist = dist.Categorical(logits=logits)
                log_probs = action_dist.log_prob(act)
                entropy = action_dist.entropy()
                entropies.append(entropy.mean().item())

                # assert probs.shape == probs_batch_tensor.shape, f'{probs.shape=}, {probs_batch_tensor.shape=}'

                actor_optimizer.zero_grad()

                ratio = (log_probs - l_p).exp()
                loss1 = ratio * adv_ext
                loss2 = torch.clip(ratio, 1. - clip_coef, 1. + clip_coef) * adv_ext

                loss = -(
                    coef_ext * torch.min(loss1, loss2).mean()
                    + coef_int * adv_int.mean()
                    + entropy_coef * entropy.mean()
                )  # we want to maximize

                losses_actor.append(loss.item())

                loss.backward()

                actor_optimizer.step()

                # optimize the critic

                critic_optimizer.zero_grad()

                critic_loss = F.mse_loss(critic(o).squeeze(-1), v_ext)

                losses_critic.append(critic_loss.item())

                critic_loss.backward()

                critic_optimizer.step()

                # optimize the intrinsic critic

                critic_int_optimizer.zero_grad()

                critic_int_loss = F.mse_loss(critic_int(next_obs).squeeze(-1), v_int)

                losses_critic_int.append(critic_int_loss.item())

                critic_int_loss.backward()

                critic_int_optimizer.step()

        for epoch in range(1):
            for o, adv_ext, v_ext, act, l_p, next_obs, v_int, adv_int in islice(dl, len(dl)): # // 4 ?

                last_frame = next_obs[:, -1:, :, :]

                fixed_random_values = fixed_random(last_frame)
                rnd_values = rnd(last_frame)

                optimizer_rnd.zero_grad()

                loss_rnd = F.mse_loss(rnd_values, fixed_random_values)
                loss_rnd.backward()

                losses_rnd.append(loss_rnd.item())

                optimizer_rnd.step()

        parcel['loss_rnd'] = sum(losses_rnd) / len(losses_rnd)
        parcel['loss_actor'] = sum(losses_actor) / len(losses_actor)
        parcel['loss_critic'] = sum(losses_critic) / len(losses_critic)
        parcel['loss_critic_int'] = sum(losses_critic_int) / len(losses_critic_int)
        parcel['entropy'] = sum(entropies) / len(entropies)

    def get_train_participants():
        statistics_collector = tp_utils.Collector(['reward'])

        def set_statistics(parcel: dict):
            if parcel.get('terminated')[0] or parcel.get('truncated')[0]:
                rewards = [x['reward'][0] for x in statistics_collector.get_entries()]
                statistics_collector.clear_entries()
                if len(rewards) > 0:
                    mean_reward = sum(rewards) / len(rewards)
                    parcel['mean_ext_reward'] = mean_reward
                parcel['episode_length'] = len(rewards)

        with (tp_utils.StepsTracker(total_timesteps, desc="steps") as steps_tracker,
              tp_tb_utils.Logging(
                 path=f"runs/{gym_environment}_rnd_{optuna_trial.datetime_start.strftime('%Y_%B_%d__%H_%M%p')}_study_{optuna_trial.study.study_name}_trial_no_{optuna_trial.number}",
                 track=[
                    'mean_ext_reward',
                    'mean_int_reward',
                    'values_ext',
                    'values_int',
                    'advantages_int'
                    'episode_length',
                    'loss_rnd',
                    'loss_actor',
                    'loss_critic',
                    'loss_critic_int',
                    'entropy',
                    'log_prob_mean',
                    'coef_int',
                    'coef_ext',
                    'entropy_coef',
                 ]) as logging):
            yield functools.partial(tp_gym_utils.call_reset, env=vec_env)
            yield steps_tracker # initialization to 0
            yield from itertools.cycle([
                    functools.partial(get_action, agent=actor),
                    functools.partial(tp_gym_utils.call_step, env=vec_env, save_obs_as="next_obs"),
                    collector,
                    statistics_collector,
                    set_statistics,
                    functools.partial(
                        tp_utils.call_after_every,
                        every_x_steps=time_steps,
                        protected=learn
                    ),
                    logging,
                    functools.partial(
                        tp_utils.call_after_every,
                        every_x_steps=5000 if should_record_video else total_timesteps + 1,
                        protected=functools.partial(record_video, videos_path=videos_path)
                    ),
                    steps_tracker, # can raise Done
                    advance,
                    functools.partial(
                        tp_gym_utils.call_reset_done, vec_env=vec_env
                    ),
            ])

    train_assembly = tp.Assembly(get_train_participants)

    train_assembly.launch()

    print(f'{learn.times_called=}')
    print(f'{learn.ns_elapsed=}')


def create_network(state_space, action_space) -> Tuple[
        StateToActionLogits,
        StateToValue,
        StateToRND,
        StateToRND_Predictor,
        StateToValue,
    ]:
    actor = StateToActionLogits(state_space, action_space)
    actor = torch.jit.script(actor)
    actor = actor.to(device)

    critic = StateToValue(state_space)
    critic = torch.jit.script(critic)
    critic = critic.to(device)

    fixed_random = StateToRND()
    for param in fixed_random.parameters():
        param.requires_grad = False
    fixed_random = torch.jit.script(fixed_random)
    fixed_random = fixed_random.to(device)

    rnd = StateToRND_Predictor() # this is refered to as "predictor" in the paper
    rnd = torch.jit.script(rnd)
    rnd = rnd.to(device)

    critic_int = StateToValue(state_space, in_channels=1)
    critic_int = torch.jit.script(critic_int)
    critic_int = critic_int.to(device)
    return actor, critic, fixed_random, rnd, critic_int


def create_network_with_optuna_trial(optuna_trial, state_space, action_space):
    return create_network(state_space, action_space)


def optuna_objective(optuna_trial):

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    gym.register_envs(ale_py)

    vec_env = SyncVectorEnv([
        functools.partial(make_env, seed=seed, max_episode_steps=18_000, repeat_action_probability=0.25) for seed in range(num_parallel_envs)
    ])

    # state and obs/observations are used in this example interchangably.

    state_space = vec_env.single_observation_space.shape[0]
    action_space = vec_env.single_action_space.n

    actor, critic, fixed_random, rnd, critic_int = create_network_with_optuna_trial(optuna_trial, state_space, action_space)

    episodes_for_evaluation = 10

    mean_reward_before_train = evaluate(vec_env.envs[0], actor, episodes_for_evaluation)
    print("before training")
    print(f'{mean_reward_before_train=}')

    total_steps = optuna_trial.suggest_int('total_timesteps', 40_000, 400_000, step=10_000)

    train(optuna_trial, vec_env, actor, critic, fixed_random, rnd, critic_int, total_timesteps=total_steps)

    mean_reward_after_train = evaluate(vec_env.envs[0], actor, episodes_for_evaluation)
    print("after training")
    print(f'{mean_reward_after_train=}')

    return mean_reward_after_train


def main():

    sqlite_file = 'optuna_trials.db'
    storage = f'sqlite:///{sqlite_file}'
    optuna_study = optuna.create_study(
        storage=storage,
        study_name=f'{gym_environment} RND - v1',
        direction="maximize",
        load_if_exists=True,
    )

    # trials = [{
    #     'total_timesteps': 40_000,
    #     'record_video': False,
    # } for _ in range(10)]
    # optuna_study.optimize(optuna_objective, n_trials=len(trials), gc_after_trial=True)
    # optuna_study.enqueue_trial(trials)

    # trials = [{
    #     'total_timesteps': 80_000,
    #     'record_video': False,
    # } for _ in range(5)]
    # optuna_study.optimize(optuna_objective, n_trials=len(trials), gc_after_trial=True)
    # optuna_study.enqueue_trial(trials)

    # trials = [{
    #     'total_timesteps': 120_000,
    #     'record_video': False,
    # } for _ in range(2)]
    # optuna_study.optimize(optuna_objective, n_trials=len(trials), gc_after_trial=True)
    # optuna_study.enqueue_trial(trials)

    optuna_study.enqueue_trial({
        'entropy_coef' : 0.001,
        'coef_int': 0.5,
        'coef_ext': 1.0,
        'total_timesteps': 400_000,
        'record_video': True,
    })
    optuna_study.optimize(optuna_objective, n_trials=1)


if __name__ == "__main__":
    main()
