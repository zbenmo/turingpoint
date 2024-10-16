from abc import ABC
from dataclasses import dataclass
from typing import Callable, Generator, Protocol, Set, Tuple, TypeVar

import numpy as np


Action = TypeVar("Action")


# forward declaration
class EnvNode(Protocol):
    ...


class EnvNode(Protocol):
    def step(self, action: Action) -> Tuple[float, EnvNode]:
        ...

    def is_done(self) -> bool:
        ...

    def available_actions(self) -> Generator[Action, None, None]:
        ...


# forward declaration
@dataclass
class TreeNode:
    ...


@dataclass
class TreeNode:
    env_node: EnvNode
    action: Action = None # Action that this TreeNode shall take if we decide to visit him (visit this TreeNode).
    parent: TreeNode = None
    children: Set[TreeNode] = None
    times_visited: int = 0
    qvalue_sum: float = 0.
    immediate_reward: float = 0. # till shown otherwise

    def is_done(self) -> bool:
        env_node = self._get_env_node()
        return env_node.is_done()
    
    def is_root(self) -> bool:
        return self.parent is None
    
    @property
    def is_leaf(self) -> bool:
        return self.children is None
    
    def get_qvalue_estimate(self) -> float:
        return self.qvalue_sum / self.times_visited if self.times_visited > 0 else 0.

    def ucb_score(self, scale=10, max_value=1e100) -> float:
        if self.times_visited == 0:
            return max_value

        if self.parent == None:
            # this is the "root"
            return max_value

        Cp = 1. / np.sqrt(2)
        N = self.parent.times_visited
        Na = self.times_visited
        U = Cp * np.sqrt((2. * np.log(N)) / Na)

        return self.get_qvalue_estimate() + scale * U

    def select_best_leaf(self) -> TreeNode:
        """Recursively desend the tree, selecting a leaf node, and returning it"""

        if self.is_leaf:
            return self

        best_child = max(self.children.values(), key=lambda c: c.ucb_score())

        return best_child.select_best_leaf()

    def expand(self) -> TreeNode:
        assert self.children is None
        self.children = {
            action: TreeNode(
                env_node=None, # access the relevant env_node throught the parent when relevant
                action=action,
                parent=self,
            )
            for action in self.env_node.available_actions()
        }
        return self.select_best_leaf()

    def _get_env_node(self):
        if self.env_node is None:
            assert self.parent.env_node is not None
            self.immediate_reward, self.env_node = self.parent.env_node.step(self.action)
        assert self.env_node is not None
        return self.env_node

    def rollout(self, gamma: float=0.99) -> float:

        # I had a thought that it is pitty that I copy here environments while this is not needed for the rollouts. TODO: !!

        env_node = self._get_env_node()
        discounted_cummulative_reward = 0
        discount_factor = 1.
        while not env_node.is_done:
            action = np.random.choice(env_node.available_actions()) # TODO: a "smarter" selection..
            reward, env_node = env_node.step(action)
            discount_factor *= gamma
            discounted_cummulative_reward += discount_factor * reward
        return discounted_cummulative_reward

    def back_propagate(self, discounted_cummulative_reward):
        self.times_visited += 1
        total_reward = self.immediate_reward + discounted_cummulative_reward 
        self.qvalue_sum += total_reward
        if self.parent is not None:
            self.parent.back_propagate(total_reward)

        # TODO: had an idea to return the rewards as a list and then to do a reduce.


class MCTS:
    """
    """

    def plan(self,
             root: TreeNode,
             *, n_iters=10_000):
        
        for _ in range(n_iters):
            selected_leaf = root.select_best_leaf()
            if selected_leaf.is_done():
                selected_leaf.back_propagate(0) # visited again, just to find out that nothing more to contribute
            else:
                selected_expanded_node = selected_leaf.expand()
                rollout_reward = selected_expanded_node.rollout()
                selected_expanded_node.back_propagate(rollout_reward)

        # A caller to this function can now select from root.children based on their qvalue_estimatimation

