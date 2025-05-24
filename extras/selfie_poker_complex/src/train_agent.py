from agent import PokerAgent
from training import Trainer

def main():
    # Initialize agent
    agent = PokerAgent()
    
    # Initialize trainer with custom parameters
    trainer = Trainer(
        agent=agent,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=10,
        checkpoint_dir="checkpoints"
    )
    
    # Start training
    print("Starting training...")
    trainer.train(num_episodes=1000, checkpoint_interval=100)
    print("Training completed!")

if __name__ == "__main__":
    main() 