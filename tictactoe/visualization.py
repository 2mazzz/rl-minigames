"""Live training dashboard for Tic Tac Toe Q-learning."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from tic_tac_toe_env import TicTacToeEnv
import time

# -- palette --
BG_DARK = "#0d1117"
BG_PANEL = "#161b22"
BG_CELL = "#1c2128"
BORDER = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED = "#8b949e"
GREEN_ACCENT = "#3fb950"
GREEN_BRIGHT = "#56d364"
GREEN_DIM = "#238636"
CYAN = "#58a6ff"
RED = "#f85149"
AMBER = "#d29922"

EMPTY_BOARD = (0, 0, 0, 0, 0, 0, 0, 0, 0)

# Shared heatmap colormap
_QCMAP = mcolors.LinearSegmentedColormap.from_list(
    "q_dark",
    [(0.0, "#da3633"), (0.5, BG_CELL), (1.0, GREEN_BRIGHT)],
)


def _style_ax(ax):
    """Apply dark theme to an axes."""
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def _draw_heatmap(ax, agent, board_state, title):
    """Render a 3x3 Q-value heatmap on the given axes."""
    ax.clear()
    ax.set_facecolor(BG_PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)

    q_values = agent.get_q_values(board_state)
    grid = q_values.reshape(3, 3)

    abs_max = max(abs(q_values.min()), abs(q_values.max()), 0.01)
    ax.imshow(
        grid,
        cmap=_QCMAP,
        vmin=-abs_max,
        vmax=abs_max,
        aspect="equal",
        origin="upper",
    )

    sym = {1: "X", 2: "O"}
    sym_color = {1: CYAN, 2: RED}
    for idx in range(9):
        r, c = divmod(idx, 3)
        val = board_state[idx]
        if val in sym:
            ax.text(
                c,
                r,
                sym[val],
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=sym_color[val],
            )
        else:
            ax.text(
                c,
                r,
                f"{q_values[idx]:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=TEXT_PRIMARY,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, fontweight="bold", pad=6)


class LiveTrainingDashboard:
    """Real-time dashboard: top row 1x2, bottom row 1x4."""

    def __init__(self):
        self.fig = None
        self.axes = {}

    def setup_figure(self):
        """Create figure with GridSpec: top 1x2, bottom 1x4."""
        plt.ion()
        plt.rcParams.update(
            {
                "text.color": TEXT_PRIMARY,
                "axes.labelcolor": TEXT_MUTED,
                "xtick.color": TEXT_MUTED,
                "ytick.color": TEXT_MUTED,
            }
        )

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.set_facecolor(BG_DARK)
        self.fig.suptitle(
            "TIC TAC TOE  Q-LEARNING",
            fontsize=15,
            fontweight="bold",
            color=GREEN_ACCENT,
            y=0.97,
        )

        gs = GridSpec(2, 4, figure=self.fig, height_ratios=[1, 1])

        # Top row — each spans 2 columns
        self.axes["board"] = self.fig.add_subplot(gs[0, 0:2])
        self.axes["curve"] = self.fig.add_subplot(gs[0, 2:4])

        # Bottom row — 4 individual heatmaps
        self.axes["hm_x_live"] = self.fig.add_subplot(gs[1, 0])
        self.axes["hm_o_live"] = self.fig.add_subplot(gs[1, 1])
        self.axes["hm_x_open"] = self.fig.add_subplot(gs[1, 2])
        self.axes["hm_o_open"] = self.fig.add_subplot(gs[1, 3])

        for ax in self.axes.values():
            _style_ax(ax)

        self.fig.tight_layout(rect=[0, 0.01, 1, 0.94])
        self.fig.subplots_adjust(hspace=0.32, wspace=0.28)
        self.refresh()

    # ------------------------------------------------------------------
    # Board rendering
    # ------------------------------------------------------------------

    def render_board(
        self, board, title=None, highlight_cells=None, highlight_color=None
    ):
        """Draw a 3x3 grid with X/O symbols on the board axes."""
        ax = self.axes["board"]
        ax.clear()
        ax.set_facecolor(BG_PANEL)
        ax.set_xlim(-0.15, 3.15)
        ax.set_ylim(-0.15, 3.15)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        # Cell backgrounds
        for idx in range(9):
            r, c = divmod(idx, 3)
            row_y = 2 - r
            rect = patches.FancyBboxPatch(
                (c + 0.04, row_y + 0.04),
                0.92,
                0.92,
                boxstyle="round,pad=0.04",
                facecolor=BG_CELL,
                edgecolor=BORDER,
                linewidth=1,
            )
            ax.add_patch(rect)

        # Highlight backgrounds (win / draw)
        if highlight_cells is not None and highlight_color is not None:
            for idx in highlight_cells:
                r, c = divmod(idx, 3)
                row_y = 2 - r
                rect = patches.FancyBboxPatch(
                    (c + 0.04, row_y + 0.04),
                    0.92,
                    0.92,
                    boxstyle="round,pad=0.04",
                    facecolor=highlight_color,
                    edgecolor=highlight_color,
                    linewidth=1.5,
                    alpha=0.35,
                )
                ax.add_patch(rect)

        # Symbols
        sym = {1: ("X", CYAN), 2: ("O", RED)}
        for idx, val in enumerate(board):
            if val in sym:
                r, c = divmod(idx, 3)
                row_y = 2 - r
                ax.text(
                    c + 0.5,
                    row_y + 0.5,
                    sym[val][0],
                    ha="center",
                    va="center",
                    fontsize=30,
                    fontweight="bold",
                    color=sym[val][1],
                )

        ax.set_title(
            title or "Live Game Board",
            color=TEXT_PRIMARY,
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

    # ------------------------------------------------------------------
    # Learning curve
    # ------------------------------------------------------------------

    def update_learning_curve(self, history):
        """Plot rolling 1000-episode win/draw rates."""
        ax = self.axes["curve"]
        ax.clear()
        ax.set_facecolor(BG_PANEL)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        winners = np.array([h["winner"] for h in history])
        window = 1000
        if len(winners) < window:
            ax.set_title(
                "Learning Curve  (waiting for data...)",
                color=TEXT_MUTED,
                fontsize=11,
                fontweight="bold",
                pad=8,
            )
            return

        kernel = np.ones(window) / window
        x_rate = np.convolve((winners == 1).astype(float), kernel, mode="valid")
        o_rate = np.convolve((winners == 2).astype(float), kernel, mode="valid")
        d_rate = np.convolve((winners == 0).astype(float), kernel, mode="valid")
        episodes = np.arange(window - 1, len(winners))

        ax.fill_between(episodes, 0, x_rate, alpha=0.12, color=CYAN)
        ax.fill_between(episodes, 0, o_rate, alpha=0.12, color=RED)
        ax.plot(episodes, x_rate, label="X win", linewidth=1.8, color=CYAN, alpha=0.9)
        ax.plot(episodes, o_rate, label="O win", linewidth=1.8, color=RED, alpha=0.9)
        ax.plot(
            episodes,
            d_rate,
            label="Draw",
            linewidth=1.4,
            color=TEXT_MUTED,
            alpha=0.7,
            linestyle="--",
        )

        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(f"Rate (rolling {window})", fontsize=9)
        ax.set_title(
            "Learning Curve", color=TEXT_PRIMARY, fontsize=11, fontweight="bold", pad=8
        )
        leg = ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.9, 1.2),
            fontsize=8,
            fancybox=True,
            framealpha=0.6,
            edgecolor=BORDER,
            borderaxespad=0,
        )
        leg.get_frame().set_facecolor(BG_DARK)
        for text in leg.get_texts():
            text.set_color(TEXT_PRIMARY)
        ax.grid(True, alpha=0.10, color=TEXT_MUTED)

    # ------------------------------------------------------------------
    # Heatmap updates
    # ------------------------------------------------------------------

    def update_live_heatmaps(self, agent_x, agent_o, board_state):
        """Update the two current-state heatmaps."""
        _draw_heatmap(self.axes["hm_x_live"], agent_x, board_state, "X  Current State")
        _draw_heatmap(self.axes["hm_o_live"], agent_o, board_state, "O  Current State")

    def update_opening_heatmaps(self, agent_x, agent_o):
        """Update the two empty-board (opening) heatmaps."""
        _draw_heatmap(self.axes["hm_x_open"], agent_x, EMPTY_BOARD, "X  Opening Values")
        _draw_heatmap(self.axes["hm_o_open"], agent_o, EMPTY_BOARD, "O  Opening Values")

    # ------------------------------------------------------------------
    # Demo game
    # ------------------------------------------------------------------

    def play_demo_game(self, env, agent_x, agent_o, episode, first_player=1):
        """Play one greedy game with step-by-step board rendering."""
        eps_x, eps_o = agent_x.epsilon, agent_o.epsilon
        agent_x.epsilon = 0.0
        agent_o.epsilon = 0.0
        agents = {1: agent_x, 2: agent_o}

        obs, info = env.reset()
        env.current_player = first_player
        info["current_player"] = first_player
        done = False
        step = 0

        board_state = env.get_state()
        starter = "X" if first_player == 1 else "O"
        self.render_board(obs, title=f"Demo  (ep {episode})  {starter} starts")
        self.update_live_heatmaps(agent_x, agent_o, board_state)
        self.update_opening_heatmaps(agent_x, agent_o)
        self.refresh()
        time.sleep(0.075)

        while not done:
            player = info["current_player"]
            state = env.get_state()
            valid = env.get_valid_actions()
            action = agents[player].choose_action(state, valid)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            board_state = env.get_state()

            result_text = ""
            if done:
                result = info.get("result", "")
                if result == "win":
                    result_text = (
                        f"  {'X' if info['current_player'] == 1 else 'O'} wins!"
                    )
                elif result == "draw":
                    result_text = "  Draw!"

            self.render_board(
                obs,
                title=f"Demo  (ep {episode})  Move {step}{result_text}",
            )
            self.update_live_heatmaps(agent_x, agent_o, board_state)
            self.refresh()
            time.sleep(0.05)

        # End-of-game highlight
        result = info.get("result", "")
        if result == "win":
            winner = info["current_player"]
            for pattern in TicTacToeEnv.WIN_PATTERNS:
                if all(obs[i] == winner for i in pattern):
                    self.render_board(
                        obs,
                        title=f"Demo  (ep {episode})  Move {step}{result_text}",
                        highlight_cells=pattern,
                        highlight_color=GREEN_BRIGHT,
                    )
                    break
        elif result == "draw":
            self.render_board(
                obs,
                title=f"Demo  (ep {episode})  Move {step}{result_text}",
                highlight_cells=list(range(9)),
                highlight_color=AMBER,
            )

        self.refresh()
        time.sleep(0.8)

        agent_x.epsilon = eps_x
        agent_o.epsilon = eps_o

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self):
        """Flush drawing updates to screen."""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
