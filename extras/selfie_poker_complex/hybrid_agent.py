import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
import pickle
import os
from agent import PokerAgent, GameObservation
from game import GameState
from mcts_agent_new import MCTSNode, MCTSAgent
from tqdm import tqdm

class HybridAgent:
    def __init__(
        self,
        exploration_weight: float = 0.5,
        max_simulations: int = 100,
        max_depth: int = 39,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        q_table_path: str = "q_table.pkl",
        reward_scale: float = 1.0
    ):
        self.exploration_weight = exploration_weight
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        self.reward_scale = reward_scale
        self.q_table = self._load_q_table()
        self.mcts = MCTSAgent(
            exploration_weight=exploration_weight,
            max_simulations=max_simulations,
            max_depth=max_depth
        )
        self.agent = PokerAgent()  # Create a PokerAgent instance for reward calculation
        self.episode_rewards = []  # Track rewards for analysis
    
    def _load_q_table(self) -> Dict[str, Dict[int, float]]:
        """Load Q-table from file if it exists, otherwise create new one"""
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'rb') as f:
                return pickle.load(f)
        return defaultdict(lambda: defaultdict(float))
    
    def _save_q_table(self):
        """Save Q-table to file"""
        with open(self.q_table_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def _get_state_key(self, state: GameObservation) -> str:
        """Convert game state to a string key for Q-table"""
        # Create a string representation of the state
        state_parts = []
        
        # Add tableau state
        for row in [state.tableau.top_row, state.tableau.middle_row, state.tableau.bottom_row]:
            row_str = ''.join(str(card) for card in row)
            state_parts.append(row_str)
        
        # Add hand state
        hand_str = ''.join(str(card) for card in state.hand)
        state_parts.append(hand_str)
        
        return '|'.join(state_parts)
    
    def get_action(self, state: GameObservation) -> int:
        """Get the best action using both Q-learning and MCTS"""
        state_key = self._get_state_key(state)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Random exploration
            valid_actions = list(range(len(state.hand)))
            return np.random.choice(valid_actions)
        
        # Get MCTS action
        mcts_action = self.mcts.get_action(state)
        
        # Get Q-value for MCTS action
        q_value = self.q_table[state_key][mcts_action]
        
        # If Q-value is too low, try to find better action
        if q_value < 0:
            valid_actions = list(range(len(state.hand)))
            best_action = max(valid_actions, key=lambda a: self.q_table[state_key][a])
            return best_action
        
        return mcts_action
    
    def update_q_value(self, state: GameObservation, action: int, reward: float, next_state: GameObservation):
        """Update Q-value based on the reward and next state"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Scale the reward
        scaled_reward = reward * self.reward_scale
        
        # Get current Q-value
        current_q = self.q_table[state_key][action]
        
        # Get maximum Q-value for next state
        next_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
        
        # Update Q-value using Q-learning formula with reward scaling
        new_q = current_q + self.learning_rate * (scaled_reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def train_episode(self, num_episodes: int = 1000, save_interval: int = 100):
        """Train the agent for multiple episodes"""
        agent = PokerAgent()
        best_reward = float('-inf')
        
        for episode in tqdm(range(num_episodes), desc="Training episodes"):
            state = agent.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done:
                # Get action from hybrid agent
                action = self.get_action(state)
                
                # Take action
                next_state, reward, done, _ = agent.step(action)
                
                # Update Q-value using the sophisticated reward from agent.py
                self.update_q_value(state, action, reward, next_state)
                
                total_reward += reward
                state = next_state
                steps += 1
            
            # Track episode rewards
            self.episode_rewards.append(total_reward)
            
            # Update best reward
            if total_reward > best_reward:
                best_reward = total_reward
                # Save best model
                self._save_q_table()
            
            # Save periodically
            if (episode + 1) % save_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-save_interval:])
                print(f"\nEpisode {episode + 1}")
                print(f"Average reward (last {save_interval} episodes): {avg_reward:.2f}")
                print(f"Best reward so far: {best_reward:.2f}")
                print(f"Steps in last episode: {steps}")
                
                # Adjust learning parameters based on performance
                if avg_reward < 0:
                    self.learning_rate *= 0.95  # Reduce learning rate if performance is poor
                    self.epsilon = min(0.5, self.epsilon * 1.05)  # Increase exploration
                elif avg_reward > 0:
                    self.epsilon = max(0.05, self.epsilon * 0.95)  # Decrease exploration
    
    def _modify_mcts_selection(self, node: MCTSNode) -> MCTSNode:
        """Modify MCTS selection to use Q-values and consider hand improvements"""
        if not node.children:
            return node
        
        # Get Q-values for current state
        state_key = self._get_state_key(node.state)
        q_values = self.q_table[state_key]
        
        # Combine UCB1 with Q-values and hand improvement potential
        def combined_value(child):
            if child.visits == 0:
                return float('inf')
            
            # Get Q-value for this action
            q_value = q_values.get(child.action, 0)
            
            # Calculate potential hand improvement reward
            potential_reward = 0
            if child.state.tableau.is_complete():
                # If tableau is complete, evaluate the round
                try:
                    score, _, is_game_over = self.agent.game_state.evaluate_round()
                    if not is_game_over:
                        potential_reward = score
                except ValueError:
                    pass
            else:
                # For incomplete tableau, calculate potential hand improvements
                potential_reward = self.agent._get_hand_upgrade_reward(node.state.tableau, child.state.tableau)
            
            # Combine MCTS value with Q-value and potential reward
            mcts_value = child.value / child.visits
            combined_value = 0.5 * mcts_value + 0.3 * q_value + 0.2 * potential_reward
            
            # Add exploration term
            exploration = self.exploration_weight * math.sqrt(math.log(node.visits) / child.visits)
            
            return combined_value + exploration
        
        # Select child with highest combined value
        return max(node.children.values(), key=combined_value) 