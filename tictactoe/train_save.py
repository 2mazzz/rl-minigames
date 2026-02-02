"""Self-play training loop that saves data for later visualization."""

import json
import gzip
import numpy as np
from pathlib import Path

from tic_tac_toe_env import TicTacToeEnv
from q_learning_agent import QLearningAgent
from evaluate import evaluate_greedy

NUM_EPISODES = 70_000
SNAPSHOT_EVERY = 500

DATA_DIR = Path(__file__).parent / "data"


def play_demo_game(env, agent_x, agent_o, first_player=1):
    """Play a greedy demo game and return the trajectory."""
    eps_x, eps_o = agent_x.epsilon, agent_o.epsilon
    agent_x.epsilon = 0.0
    agent_o.epsilon = 0.0
    agents = {1: agent_x, 2: agent_o}

    obs, info = env.reset()
    env.current_player = first_player
    info["current_player"] = first_player

    trajectory = []
    done = False

    while not done:
        player = info["current_player"]
        state = env.get_state()
        valid = env.get_valid_actions()
        action = agents[player].choose_action(state, valid)

        trajectory.append({
            "board": [int(x) for x in state],
            "player": player,
            "action": int(action),
        })

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Final state and result
    trajectory.append({
        "board": [int(x) for x in env.get_state()],
        "result": info.get("result", ""),
        "winner": int(info.get("current_player")) if info.get("result") == "win" else 0,
    })

    agent_x.epsilon = eps_x
    agent_o.epsilon = eps_o
    return trajectory


def train():
    DATA_DIR.mkdir(exist_ok=True)

    env = TicTacToeEnv()
    agent_x = QLearningAgent(player_id=1)
    agent_o = QLearningAgent(player_id=2)
    agents = {1: agent_x, 2: agent_o}

    # Data to save
    episode_results = []  # 0=draw, 1=X win, 2=O win
    snapshots = []  # periodic snapshots with Q-tables and demo games
    demo_count = 0

    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        if episode % 2 == 0:
            env.current_player = 2
            info["current_player"] = 2
        done = False

        last_transition = {1: None, 2: None}

        while not done:
            player = info["current_player"]
            agent = agents[player]
            state = env.get_state()
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = env.get_state()

            opponent = 3 - player
            if last_transition[opponent] is not None:
                lt = last_transition[opponent]
                if done:
                    result = info.get("result", "")
                    if result == "win":
                        opp_reward = -1.0
                    elif result == "draw":
                        opp_reward = 0.5
                    else:
                        opp_reward = 0.0
                    agents[opponent].update(
                        lt["state"], lt["action"], opp_reward, next_state, True
                    )
                else:
                    agents[opponent].update(
                        lt["state"], lt["action"], 0.0, next_state, False
                    )
                last_transition[opponent] = None

            if done:
                agent.update(state, action, reward, next_state, True)
                result = info.get("result", "")
                if result == "win":
                    winner = player
                elif result == "draw":
                    winner = 0
                else:
                    winner = 3 - player
            else:
                last_transition[player] = {"state": state, "action": action}

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

        episode_results.append(winner)

        # Periodic snapshot with Q-tables and demo game
        if episode % SNAPSHOT_EVERY == 0:
            first_player = 1 if demo_count % 2 == 0 else 2
            demo_game = play_demo_game(env, agent_x, agent_o, first_player)
            demo_count += 1

            snapshots.append({
                "episode": episode,
                "epsilon": agent_x.epsilon,
                "q_table_x": {str(k): v.tolist() for k, v in agent_x.q_table.items()},
                "q_table_o": {str(k): v.tolist() for k, v in agent_o.q_table.items()},
                "demo_game": demo_game,
            })
            print(f"Episode {episode:>7,} | Epsilon: {agent_x.epsilon:.4f} | "
                  f"Q-tables: X={len(agent_x.q_table)}, O={len(agent_o.q_table)}")

    # Save episode results
    with open(DATA_DIR / "episode_results.json", "w") as f:
        json.dump(episode_results, f)

    # Save snapshots (compressed - large file)
    print("Saving snapshots (compressed)...")
    with gzip.open(DATA_DIR / "snapshots.json.gz", "wt", encoding="utf-8") as f:
        json.dump(snapshots, f)

    # Greedy evaluation
    print("Running greedy evaluation...")
    stats = evaluate_greedy(agent_x, agent_o, env)
    with open(DATA_DIR / "evaluation.json", "w") as f:
        json.dump(stats, f, indent=2)

    total = stats["x_wins"] + stats["o_wins"] + stats["draws"]
    print(f"Greedy results ({total} games): "
          f"X={stats['x_wins']} O={stats['o_wins']} Draws={stats['draws']}")
    print(f"\nData saved to {DATA_DIR}/")


if __name__ == "__main__":
    train()
