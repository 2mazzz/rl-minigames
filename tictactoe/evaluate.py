"""Evaluation and plotting utilities for Tic Tac Toe Q-learning."""

import os
import numpy as np
import matplotlib.pyplot as plt


def evaluate_greedy(agent_x, agent_o, env, n_games=10000):
    """Run games with epsilon=0 and return result counts.

    Returns dict with keys: x_wins, o_wins, draws.
    """
    # Save and override epsilon
    eps_x, eps_o = agent_x.epsilon, agent_o.epsilon
    agent_x.epsilon = 0.0
    agent_o.epsilon = 0.0

    agents = {1: agent_x, 2: agent_o}
    results = {"x_wins": 0, "o_wins": 0, "draws": 0}

    for game_i in range(n_games):
        obs, info = env.reset()
        # Alternate who starts: even games X, odd games O
        if game_i % 2 == 1:
            env.current_player = 2
            info["current_player"] = 2
        done = False
        while not done:
            player = info["current_player"]
            agent = agents[player]
            state = tuple(obs)
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        result = info.get("result", "")
        if result == "win":
            if info["current_player"] == 1:
                results["x_wins"] += 1
            else:
                results["o_wins"] += 1
        elif result == "draw":
            results["draws"] += 1
        elif result == "invalid":
            # Count invalid moves as a loss for the player who made them
            if info["current_player"] == 1:
                results["o_wins"] += 1
            else:
                results["x_wins"] += 1

    # Restore epsilon
    agent_x.epsilon = eps_x
    agent_o.epsilon = eps_o

    return results


def plot_learning_curve(history, save_dir="figures"):
    """Plot rolling win/draw rates over training episodes.

    history: list of dicts with keys 'winner' (1, 2, or 0 for draw).
    """
    os.makedirs(save_dir, exist_ok=True)

    winners = np.array([h["winner"] for h in history])
    window = 1000

    if len(winners) < window:
        print("Not enough episodes to plot learning curve.")
        return

    x_wins = (winners == 1).astype(float)
    o_wins = (winners == 2).astype(float)
    draws = (winners == 0).astype(float)

    # Rolling averages
    kernel = np.ones(window) / window
    x_rate = np.convolve(x_wins, kernel, mode="valid")
    o_rate = np.convolve(o_wins, kernel, mode="valid")
    draw_rate = np.convolve(draws, kernel, mode="valid")

    episodes = np.arange(window - 1, len(winners))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, x_rate, label="X win rate", alpha=0.8)
    plt.plot(episodes, o_rate, label="O win rate", alpha=0.8)
    plt.plot(episodes, draw_rate, label="Draw rate", alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel(f"Rate (rolling {window})")
    plt.title("Learning Curve — Self-Play Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_curve.png"), dpi=150)
    plt.close()
    print(f"Saved learning curve to {save_dir}/learning_curve.png")


def plot_final_stats(stats, save_dir="figures"):
    """Bar chart of greedy evaluation results."""
    os.makedirs(save_dir, exist_ok=True)

    labels = ["X wins", "O wins", "Draws"]
    values = [stats["x_wins"], stats["o_wins"], stats["draws"]]
    total = sum(values)
    percentages = [v / total * 100 for v in values]

    colors = ["#e74c3c", "#3498db", "#95a5a6"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, percentages, color=colors, edgecolor="black", linewidth=0.5)

    for bar, pct, val in zip(bars, percentages, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{pct:.1f}%\n({val})",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.ylabel("Percentage (%)")
    plt.title(f"Greedy Evaluation Results ({total} games)")
    plt.ylim(0, max(percentages) + 15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "greedy_evaluation.png"), dpi=150)
    plt.close()
    print(f"Saved greedy evaluation to {save_dir}/greedy_evaluation.png")
