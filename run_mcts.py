import os
from mcts_agent_new import MCTSAgent, train_mcts_agent

def main():
    print("Starting MCTS agent training...")
    print("This will run 1000 episodes")
    print("Training progress will be shown every 10 episodes")
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "mcts_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train the agent
    agent = train_mcts_agent(
        num_episodes=1000,
        save_interval=100,
        checkpoint_dir=checkpoint_dir
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