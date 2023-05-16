import argparse
from copy import deepcopy
import os
from typing import Optional, Tuple

import gymnasium as gym
from pettingzoo.classic import tictactoe_v3
from dataclasses import dataclass, field
import heapq
from turingpoint.self_play import SelfPlay, Quality, Agent

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass(order=True)
class PriorityQueueElement:
  quality: Quality
  agent: Agent = field(compare=False)


class PriorityQueue:
  """
  Maintains at most max_items. Smaller items leave first.
  """
  def __init__(self, max_items: int):
    self.max_items = max_items
    self.heap = []

  def push(self, element: PriorityQueueElement):
    """
    Adds to the heap, maintaining heap property.
    """
    if len(self.heap) < self.max_items:
      heapq.heappush(self.heap, element)
    else:
      _ = heapq.heappushpop(self.heap, element)

  def pop(self) -> PriorityQueueElement:
    """
    With current implementation it pops and returns the smallest item.
    """
    return heapq.heappop(self.heap)

  def empty(self) -> bool:
    return len(self.heap) < 1


def get_env(render_mode=None):
    return PettingZooEnv(tictactoe_v3.env(render_mode=render_mode))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
    )
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument(
        '--win-rate',
        type=float,
        default=0.6,
        help='the expected winning rate: Optimal policy can get 0.7'
    )
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    parser.add_argument(
        '--agent-id',
        type=int,
        default=2,
        help='the learned agent plays as the'
        ' agent_id-th player. Choices are 1 and 2.'
    )
    parser.add_argument(
        '--resume-path',
        type=str,
        default='',
        help='the path of agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--opponent-path',
        type=str,
        default='',
        help='the path of opponent agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def self_play_tic_tac_toe(
  args: argparse.Namespace = get_args(),
  agent_learn: Optional[BasePolicy] = None,
  agent_opponent: Optional[BasePolicy] = None,
  optim: Optional[torch.optim.Optimizer] = None
):

  class TicTacToeSelfPlay(SelfPlay):
    def __init__(self):

      # ======== environment setup =========
      self.train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
      self.test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
      # seed
      np.random.seed(args.seed)
      torch.manual_seed(args.seed)
      self.train_envs.seed(args.seed)
      self.test_envs.seed(args.seed)

      assert agent_learn == None
      assert agent_opponent == None
      assert args.agent_id == 2

      # ======== agent setup =========
      policy, self.optim, self.agents = self._get_agents(
          args, agent_learn=agent_learn, agent_opponent=agent_opponent, optim=optim
      )

      assert isinstance(policy, MultiAgentPolicyManager)
      assert len(self.agents) == 2
      assert self.agents[0] == "player_1", f'{self.agents[0]=}'
      assert self.agents[1] == "player_2", f'{self.agents[1]=}'

      self.agent = policy.policies[self.agents[1]]

      save_best_maximum_count = 10
      self.agents_queue = PriorityQueue(save_best_maximum_count)

      initial_ELO = 1200
      self.agents_queue.push(PriorityQueueElement(initial_ELO, deepcopy(self.agent)))

      self.agent_under_training = None
      self.rewards = None

    # ======== callback functions used during training =========
    def _save_best_fn(self, policy):
        if hasattr(args, 'model_save_path'):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, 'tic_tac_toe', 'dqn', 'policy.pth'
            )
        torch.save(
            policy.policies[self.agents[args.agent_id - 1]].state_dict(), model_save_path
        )

    def _stop_fn(self, mean_rewards):
        return mean_rewards >= args.win_rate

    def _train_fn(self, epoch, env_step):
        self.agent.set_eps(args.eps_train)

    def _test_fn(self, epoch, env_step):
        self.agent.set_eps(args.eps_test)

    def _reward_metric(self, rews):
        return rews[:, self.agent_under_training]

    def fetch_agent_to_train(self) -> Agent:
      return self.agent

    def save_agent(self, agent):
    #   self._save_best_fn(self.policy)
      current_ELO = 1201 # TODO: ???
      self.agents_queue.push(PriorityQueueElement(current_ELO, deepcopy(self.agent)))
      # at the moment, I'm not saving here but only updating the priority queue

    def fetch_opponent(self) -> Agent:
      assert not self.agents_queue.empty()
      opponent_entry = np.random.choice(self.agents_queue.heap)
      return opponent_entry.agent

    def _get_agents(self,
        args: argparse.Namespace = get_args(),
        agent_learn: Optional[BasePolicy] = None,
        agent_opponent: Optional[BasePolicy] = None,
        optim: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
        env = get_env()
        observation_space = env.observation_space['observation'] if isinstance(
            env.observation_space, gym.spaces.Dict
        ) else env.observation_space
        args.state_shape = observation_space.shape or observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n
        if agent_learn is None:
            # model
            net = Net(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device
            ).to(args.device)
            if optim is None:
                optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            agent_learn = DQNPolicy(
                net,
                optim,
                args.gamma,
                args.n_step,
                target_update_freq=args.target_update_freq
            )
            if args.resume_path:
                agent_learn.load_state_dict(torch.load(args.resume_path))

        if agent_opponent is None:
            if args.opponent_path:
                agent_opponent = deepcopy(agent_learn)
                agent_opponent.load_state_dict(torch.load(args.opponent_path))
            else:
                agent_opponent = RandomPolicy()

        if args.agent_id == 1:
            agents = [agent_learn, agent_opponent]
        else:
            agents = [agent_opponent, agent_learn]
        policy = MultiAgentPolicyManager(agents, env)
        return policy, optim, env.agents

    def train_against_agent(self, agent_to_train: Agent, opponent_agent: Agent):
      self.rewards = []
      for game in tqdm(range(10), desc="games", total=10, leave=False):
        self.agent_under_training = 0 if game % 2 == 0 else 1
        agents = [agent_to_train, opponent_agent] if self.agent_under_training == 0 else [opponent_agent, agent_to_train]
        env = get_env()
        policy = MultiAgentPolicyManager(agents, env)

        # ======== collector setup =========
        train_collector = Collector(
            policy,
            self.train_envs,
            VectorReplayBuffer(args.buffer_size, len(self.train_envs)),
            exploration_noise=True
        )
        test_collector = Collector(policy, self.test_envs, exploration_noise=True)
        # policy.set_eps(1)
        train_collector.collect(n_step=args.batch_size * args.training_num)

        # ======== tensorboard logging setup =========
        log_path = os.path.join(args.logdir, 'tic_tac_toe', 'dqn')
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        self.logger = TensorboardLogger(writer)

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            10, # args.epoch,
            10, # args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            train_fn=self._train_fn,
            test_fn=self._test_fn,
            stop_fn=self._stop_fn,
            save_best_fn=self._save_best_fn,
            update_per_step=args.update_per_step,
            logger=self.logger,
            test_in_train=False,
            reward_metric=self._reward_metric,
            verbose=False,
            show_progress=False
        )

        # best_reward = result["best_reward"]

        # self.rewards.append(best_reward if self.agent_under_training == 0 else -best_reward)

    def evaluate_agent(self, agent: Agent) -> Quality:
      """Evaluates against a RandomPolicy.
      10 times as the first player,
      10 times as the second player.

      Args:
        agent: the agent to evaluate

      Returns:
        (the Quality) is the mean of the rewards for the 10+10=20 games.
      """
      env = get_env()

      rewards = []

      policy = MultiAgentPolicyManager([agent, RandomPolicy()], env)
      policy.eval()
      test_collector = Collector(policy, self.test_envs, exploration_noise=True)
      result = test_collector.collect(n_episode=10, render=False)
      rewards.extend(rew[0] for rew in result['rews'])

      policy = MultiAgentPolicyManager([RandomPolicy(), agent], env)
      policy.eval()
      test_collector = Collector(policy, self.test_envs, exploration_noise=True)
      result = test_collector.collect(n_episode=10, render=False)
      rewards.extend(rew[1] for rew in result['rews'])
      
      return np.mean(rewards)

      # 1200 # TODO: self.reward_metric() # TODO: at the moment there is a missing argument: rew

  tic_tac_toe_self_play = TicTacToeSelfPlay()

  pbar = tqdm(total=100, desc="rounds", position=0, leave=True)

  rounds = 0
  def should_stop():
    nonlocal rounds
    nonlocal pbar

    rounds += 1
    pbar.update(1)
    return rounds >= 100

  mean_rewards = []

  def should_save():
    nonlocal mean_rewards

    mean_rewards.append(tic_tac_toe_self_play.evaluate_agent(tic_tac_toe_self_play.agent))

    return True

  tic_tac_toe_self_play.launch(should_stop=should_stop, should_save=should_save)

  def plot_mean_rewards_over_time(mean_rewards):
    x = list(range(len(mean_rewards)))
    plt.plot(x, mean_rewards, 'o')
    m, b = np.polyfit(x, mean_rewards, 1)
    plt.plot(x, np.multiply(m, x) + b)
    plt.show()

  plot_mean_rewards_over_time(mean_rewards)


if __name__ == "__main__":
  args = get_args()
  self_play_tic_tac_toe(args)