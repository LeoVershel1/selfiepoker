import numpy as np
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import json
from datetime import datetime
from agent import PokerAgent, GameObservation

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class Trainer:
    def __init__(
        self,
        agent: PokerAgent,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10,
        checkpoint_dir: str = "checkpoints"
    ):
        self.agent = agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQN(agent.observation_space, agent.action_space).to(self.device)
        self.target_net = DQN(agent.observation_space, agent.action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        # Training parameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start  # Current epsilon value
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = float('-inf')
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create metadata file for this training run
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata_path = os.path.join(checkpoint_dir, f"metadata_{self.run_id}.json")
        self._save_metadata()
    
    def _save_metadata(self):
        """Save training configuration and hyperparameters"""
        metadata = {
            "run_id": self.run_id,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "buffer_size": self.memory.buffer.maxlen,
            "batch_size": self.batch_size,
            "target_update": self.target_update,
            "device": str(self.device),
            "model_architecture": str(self.policy_net)
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def save_checkpoint(self, episode: int, is_best: bool = False):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'best_reward': self.best_reward
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.run_id}_ep{episode}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"best_model_{self.run_id}.pth")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.best_reward = checkpoint['best_reward']
        return checkpoint['episode']
    
    def get_best_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Get the best action and its Q-value for a given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            # Mask invalid actions
            valid_actions = self.agent.get_valid_actions()
            mask = torch.ones_like(q_values) * float('-inf')
            mask[0, valid_actions] = 0
            q_values = q_values + mask
            
            # Get best action and its Q-value
            best_action = q_values.max(1)[1].item()
            best_q_value = q_values.max(1)[0].item()
            
            return best_action, best_q_value
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Exploration: choose random action from valid actions
            return random.choice(self.agent.get_valid_actions())
        
        # Exploitation: choose best action according to policy network
        best_action, _ = self.get_best_action(state)
        return best_action
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from replay buffer
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes: int, checkpoint_interval: int = 100):
        for episode in range(num_episodes):
            observation = self.agent.reset()
            state = self.agent.get_state_representation()  # Convert to numerical representation
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select and perform action
                action = self.select_action(state, training=True)
                next_observation, reward, done, _ = self.agent.step(action)
                next_state = self.agent.get_state_representation()  # Convert to numerical representation
                
                # Store transition in replay buffer
                self.memory.push(state, action, reward, next_state, done)
                
                # Move to next state
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Perform one step of optimization
                loss = self.optimize_model()
                
                if done:
                    break
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Update best reward and save checkpoint if needed
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_checkpoint(episode, is_best=True)
            
            # Save regular checkpoint
            if episode % checkpoint_interval == 0:
                self.save_checkpoint(episode)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {episode}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Average Length: {avg_length:.2f}")
                print(f"Epsilon: {self.epsilon:.3f}")
                print(f"Best Reward: {self.best_reward:.2f}")
                if loss is not None:
                    print(f"Loss: {loss:.4f}")
                print("-------------------")

class PokerModel:
    """Wrapper class for using trained model in evaluation mode"""
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        self.model = DQN(
            input_size=checkpoint['policy_net_state_dict']['network.0.weight'].size(1),
            output_size=checkpoint['policy_net_state_dict']['network.4.weight'].size(0)
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['policy_net_state_dict'])
        self.model.eval()  # Set to evaluation mode
    
    def get_best_action(self, state: np.ndarray, valid_actions: List[int]) -> Tuple[int, float]:
        """Get the best action and its Q-value for a given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            
            # Mask invalid actions
            mask = torch.ones_like(q_values) * float('-inf')
            mask[0, valid_actions] = 0
            q_values = q_values + mask
            
            # Get best action and its Q-value
            best_action = q_values.max(1)[1].item()
            best_q_value = q_values.max(1)[0].item()
            
            return best_action, best_q_value

if __name__ == "__main__":
    # Initialize agent and trainer
    agent = PokerAgent()
    trainer = Trainer(agent)
    
    # Train the agent
    trainer.train(num_episodes=1000, checkpoint_interval=100)
    
    # Example of how to use the trained model
    model = PokerModel("checkpoints/best_model_latest.pth")
    state = agent.get_state_representation()
    valid_actions = agent.get_valid_actions()
    best_action, q_value = model.get_best_action(state, valid_actions)
    print(f"Best action: {best_action}, Q-value: {q_value:.2f}") 