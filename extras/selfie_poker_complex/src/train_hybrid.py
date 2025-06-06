from hybrid_agent import HybridAgent
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train hybrid Q-learning + MCTS agent')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--save-interval', type=int, default=100, help='Interval for saving Q-table')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Q-learning learning rate')
    parser.add_argument('--discount', type=float, default=0.99, help='Q-learning discount factor')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Initial exploration rate')
    parser.add_argument('--mcts-simulations', type=int, default=200, help='Number of MCTS simulations')
    parser.add_argument('--q-table', type=str, default='q_table.pkl', help='Path to save/load Q-table')
    parser.add_argument('--reward-scale', type=float, default=0.1, help='Scale factor for rewards')
    
    args = parser.parse_args()
    
    # Create hybrid agent
    agent = HybridAgent(
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        epsilon=args.epsilon,
        max_simulations=args.mcts_simulations,
        q_table_path=args.q_table,
        reward_scale=args.reward_scale
    )
    
    # Train agent
    print(f"Starting training for {args.episodes} episodes...")
    print(f"Parameters:")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Discount factor: {args.discount}")
    print(f"- Initial epsilon: {args.epsilon}")
    print(f"- MCTS simulations: {args.mcts_simulations}")
    print(f"- Reward scale: {args.reward_scale}")
    agent.train_episode(num_episodes=args.episodes, save_interval=args.save_interval)
    print("Training complete!")

if __name__ == "__main__":
    main() 