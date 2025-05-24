import os
import argparse
from mcts_agent_new import MCTSAgent, train_mcts_agent

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MCTS agent for poker game')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--simulations', type=int, default=1000, help='Number of MCTS simulations per action')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--checkpoint-dir', type=str, default='mcts_checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()

    print("Starting MCTS agent training...")
    print(f"This will run {args.episodes} episodes")
    print(f"Using {args.workers or 'CPU count - 1'} worker processes")
    print("Training progress will be shown every 10 episodes")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Train the agent
    agent = train_mcts_agent(
        num_episodes=args.episodes,
        save_interval=100,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.workers
    )
    
    print("\nTraining complete!")
    print("Testing the trained agent...")
    
    # Test the trained agent
    state = agent.agent.reset()
    total_reward = 0
    step_count = 0
    
    while not agent.agent.game_state.check_game_over():
        step_count += 1
        print(f"\nStep {step_count}")
        
        # Get action from MCTS
        action = agent.get_action(state)
        print(f"MCTS chose action {action}")
        
        # Take the action
        next_state, reward, done, _ = agent.agent.step(action)
        print(f"Step reward: {reward}, done: {done}")
        
        # Update state and reward
        state = next_state
        total_reward += reward
        print(f"Total reward so far: {total_reward}")
    
    print(f"\nTest complete!")
    print(f"Total steps: {step_count}")
    print(f"Final reward: {total_reward}")

if __name__ == "__main__":
    main() 