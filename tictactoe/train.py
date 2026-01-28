"""Self-play training loop for Tic Tac Toe Q-learning agents."""

from tic_tac_toe_env import TicTacToeEnv
from q_learning_agent import QLearningAgent
from evaluate import evaluate_greedy, plot_learning_curve, plot_final_stats

NUM_EPISODES = 100_000
PRINT_EVERY = 10_000


def train():
    env = TicTacToeEnv()
    agent_x = QLearningAgent(player_id=1)
    agent_o = QLearningAgent(player_id=2)
    agents = {1: agent_x, 2: agent_o}

    history = []
    rolling_stats = {"x_wins": 0, "o_wins": 0, "draws": 0}

    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        # Alternate who starts: even episodes X, odd episodes O
        if episode % 2 == 0:
            env.current_player = 2
            info["current_player"] = 2
        done = False

        # Store last transition per player for delayed reward
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

            # Update the opponent's previous transition now that they can
            # see the board after the current player moved
            opponent = 3 - player
            if last_transition[opponent] is not None:
                lt = last_transition[opponent]
                if done:
                    result = info.get("result", "")
                    if result == "win":
                        opp_reward = -1.0  # current player won, opponent lost
                    elif result == "draw":
                        opp_reward = 0.5
                    else:
                        opp_reward = 0.0
                    agents[opponent].update(
                        lt["state"], lt["action"], opp_reward, next_state, True
                    )
                else:
                    # Game continues — opponent gets 0 reward, sees new board state
                    agents[opponent].update(
                        lt["state"], lt["action"], 0.0, next_state, False
                    )
                last_transition[opponent] = None

            if done:
                # Current player gets their reward directly
                agent.update(state, action, reward, next_state, True)

                # Record result
                result = info.get("result", "")
                if result == "win":
                    winner = player
                    rolling_stats["x_wins" if player == 1 else "o_wins"] += 1
                elif result == "draw":
                    winner = 0
                    rolling_stats["draws"] += 1
                else:
                    # Invalid move = loss for current player
                    winner = 3 - player
                    rolling_stats["x_wins" if winner == 1 else "o_wins"] += 1
            else:
                # Store transition for delayed update on next opponent move
                last_transition[player] = {
                    "state": state,
                    "action": action,
                }

        # Decay epsilon for both agents
        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

        history.append({"winner": winner})

        # Print rolling stats
        if episode % PRINT_EVERY == 0:
            total = sum(rolling_stats.values())
            print(
                f"Episode {episode:>7,} | "
                f"X wins: {rolling_stats['x_wins']/total*100:5.1f}% | "
                f"O wins: {rolling_stats['o_wins']/total*100:5.1f}% | "
                f"Draws: {rolling_stats['draws']/total*100:5.1f}% | "
                f"Epsilon: {agent_x.epsilon:.4f} | "
                f"Q-table sizes: X={len(agent_x.q_table)}, O={len(agent_o.q_table)}"
            )
            rolling_stats = {"x_wins": 0, "o_wins": 0, "draws": 0}

    print("\nTraining complete. Running greedy evaluation...")
    stats = evaluate_greedy(agent_x, agent_o, env)
    total = stats["x_wins"] + stats["o_wins"] + stats["draws"]
    print(
        f"Greedy results ({total} games): "
        f"X wins: {stats['x_wins']} ({stats['x_wins']/total*100:.1f}%) | "
        f"O wins: {stats['o_wins']} ({stats['o_wins']/total*100:.1f}%) | "
        f"Draws: {stats['draws']} ({stats['draws']/total*100:.1f}%)"
    )

    print("\nGenerating plots...")
    plot_learning_curve(history)
    plot_final_stats(stats)
    print("Done!")


if __name__ == "__main__":
    train()
