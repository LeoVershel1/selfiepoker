from mcts_agent import train_mcts_agent, load_trained_agent
import os

def main():
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = "mcts_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Starting MCTS agent training...")
    print("This will run 1000 episodes with checkpoints every 100 episodes")
    print("Training progress will be shown every 10 episodes")
    
    # Train the agent
    agent = train_mcts_agent(
        num_episodes=10000,
        save_interval=100,
        checkpoint_dir=checkpoint_dir
    )
    
    print("\nTraining complete!")
    print("Testing the trained agent...")
    
    # Test the agent
    state = agent.agent.reset()
    total_reward = 0
    moves = 0
    
    while not agent.agent.game_state.check_game_over():
        action = agent.get_action(state)
        state, reward, done, _ = agent.agent.step(action)
        total_reward += reward
        moves += 1
        print(f"Move {moves}: Action {action}, Reward: {reward:.2f}")
    
    print(f"\nTest game complete!")
    print(f"Total moves: {moves}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per move: {total_reward/moves:.2f}")

if __name__ == "__main__":
    main() 