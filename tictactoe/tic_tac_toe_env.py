"""Gymnasium-compatible Tic Tac Toe environment for self-play."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TicTacToeEnv(gym.Env):
    """Two-player Tic Tac Toe environment.

    Board values: 0=empty, 1=player X, 2=player O.
    Players alternate turns, starting with player 1 (X).
    """

    metadata = {"render_modes": ["human"]}

    # Win patterns: rows, columns, diagonals
    WIN_PATTERNS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6],              # diagonals
    ]

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.MultiDiscrete([3] * 9)
        self.action_space = spaces.Discrete(9)
        self.render_mode = render_mode
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # 1 = X, 2 = O

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1
        return self.board.copy(), {"current_player": self.current_player}

    def step(self, action):
        # Invalid move: cell already occupied
        if self.board[action] != 0:
            return (
                self.board.copy(),
                -1.0,
                True,
                False,
                {"current_player": self.current_player, "result": "invalid"},
            )

        # Place mark
        self.board[action] = self.current_player

        # Check win
        if self._check_win(self.current_player):
            return (
                self.board.copy(),
                1.0,
                True,
                False,
                {"current_player": self.current_player, "result": "win"},
            )

        # Check draw
        if not np.any(self.board == 0):
            return (
                self.board.copy(),
                0.5,
                True,
                False,
                {"current_player": self.current_player, "result": "draw"},
            )

        # Game continues — switch player
        self.current_player = 3 - self.current_player
        return (
            self.board.copy(),
            0.0,
            False,
            False,
            {"current_player": self.current_player},
        )

    def _check_win(self, player):
        for pattern in self.WIN_PATTERNS:
            if all(self.board[i] == player for i in pattern):
                return True
        return False

    def get_valid_actions(self):
        return np.where(self.board == 0)[0]

    def get_state(self):
        """Return hashable state for Q-table lookup."""
        return tuple(self.board)

    def render(self):
        if self.render_mode != "human":
            return
        symbols = {0: ".", 1: "X", 2: "O"}
        for row in range(3):
            print(" ".join(symbols[self.board[row * 3 + col]] for col in range(3)))
        print()
