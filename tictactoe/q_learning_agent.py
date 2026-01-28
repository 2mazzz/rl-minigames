"""Tabular Q-learning agent for Tic Tac Toe."""

import numpy as np


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning agent.

    Q-table is a dict mapping board state tuples to numpy arrays of size 9
    (one value per action).
    """

    def __init__(
        self,
        player_id,
        lr=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99995,
    ):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[tuple, np.ndarray] = {}

    def _get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        return self.q_table[state]

    def choose_action(self, state, valid_actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        q_values = self._get_q_values(state)
        # Among valid actions, pick the one with highest Q-value (random tiebreak)
        valid_q = q_values[valid_actions]
        max_q = np.max(valid_q)
        best_actions = valid_actions[valid_q == max_q]
        return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        q_values = self._get_q_values(state)
        if done:
            target = reward
        else:
            next_q = self._get_q_values(next_state)
            target = reward + self.gamma * np.max(next_q)
        q_values[action] += self.lr * (target - q_values[action])

    def get_q_values(self, state):
        """Return a copy of the full 9-element Q-value array (read-only)."""
        if state in self.q_table:
            return self.q_table[state].copy()
        return np.zeros(9)

    def get_state_value(self, state):
        """Return max Q-value for a state, or 0.0 if unseen."""
        if state in self.q_table:
            return float(np.max(self.q_table[state]))
        return 0.0

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
