import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
from agent import PokerAgent, GameObservation
from game import GameState
import multiprocessing as mp
from functools import partial
import os
import time
from tqdm import tqdm

class MCTSNode:
    def __init__(self, state: GameObservation, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children: Dict[int, MCTSNode] = {}  # action -> node
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None  # Will be initialized when needed

class MCTSAgent:
    def __init__(
        self,
        exploration_weight: float = 0.5, 
        max_simulations: int = 100,
        max_depth: int = 39,
        num_workers: int = None
    ):
        self.exploration_weight = exploration_weight
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.agent = PokerAgent()
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
    
    def get_action(self, state: GameObservation) -> int:
        """Get the best action for the current state using MCTS"""
        # Create root node
        root = MCTSNode(state)
        
        # Run simulations sequentially since we're already in a worker process
        for _ in range(self.max_simulations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.state.tableau.is_complete():
                node = self._expand(node)
            
            # Simulation
            value = self._simulate(node)
            
            # Backpropagation
            self._backpropagate(node, value)
        
        # Choose best action
        if root.children:
            # Choose action with highest visit count
            best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
            return best_action
        
        return 0
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1"""
        # If this is a leaf node or we have untried actions, expand it
        if not node.children or (node.untried_actions is not None and node.untried_actions):
            return self._expand(node)
        
        # Otherwise, traverse down the tree
        current = node
        while current.children and not current.state.tableau.is_complete():
            # If we have untried actions, expand
            if current.untried_actions and current.untried_actions:
                return self._expand(current)
            
            # Select child using UCB1
            current = self._ucb_select(current)
        
        return current
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand the node by trying an untried action"""
        # Initialize untried actions if not already done
        if node.untried_actions is None:
            # Get valid actions - just the indices of cards in hand
            node.untried_actions = list(range(len(node.state.hand)))
        
        if not node.untried_actions:
            return node
        
        # Get the next untried action
        action = node.untried_actions.pop()
        
        # Create a new agent instance for this expansion
        expand_agent = PokerAgent()
        expand_agent.game_state = GameState()
        # Copy tableau rows properly
        expand_agent.game_state.tableau.top_row = node.state.tableau.top_row[:]
        expand_agent.game_state.tableau.middle_row = node.state.tableau.middle_row[:]
        expand_agent.game_state.tableau.bottom_row = node.state.tableau.bottom_row[:]
        expand_agent.game_state.hand = node.state.hand[:]
        
        # Take the action
        next_state, reward, done, _ = expand_agent.step(action)
        
        # Create new node with the next state
        child = MCTSNode(next_state, parent=node, action=action)
        node.children[action] = child
        
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """Simulate a playout from the node using a simple strategy"""
        # Create a new agent instance for this simulation
        sim_agent = PokerAgent()
        sim_agent.game_state = GameState()
        # Copy tableau rows properly
        sim_agent.game_state.tableau.top_row = node.state.tableau.top_row[:]
        sim_agent.game_state.tableau.middle_row = node.state.tableau.middle_row[:]
        sim_agent.game_state.tableau.bottom_row = node.state.tableau.bottom_row[:]
        sim_agent.game_state.hand = node.state.hand[:]
        
        state = node.state
        depth = 0
        total_reward = 0
        discount = 1.0
        
        while not state.tableau.is_complete() and depth < self.max_depth:
            # Get valid actions
            valid_actions = sim_agent.get_valid_actions()
            if not valid_actions:
                break
            
            # Choose action based on simple heuristic instead of random
            action = self._choose_simulation_action(sim_agent, valid_actions)
            
            # Take action and get reward from the agent's reward system
            next_state, reward, done, _ = sim_agent.step(action)
            
            # Use only the agent's reward, with discount
            total_reward += reward * discount
            
            # Update state for next iteration
            state = next_state
            discount *= 0.98 
            depth += 1
            
            if done:
                break
        
        return total_reward
    
    def _choose_simulation_action(self, agent: PokerAgent, valid_actions: List[int]) -> int:
        """Choose an action during simulation using a simple heuristic"""
        if not valid_actions:
            return 0
            
        # Get current state
        state = agent.game_state
        
        # Try to find actions that improve hand strength
        best_action = valid_actions[0]
        best_reward = float('-inf')
        
        for action in valid_actions:
            # Create a copy of the state to evaluate the action
            temp_agent = PokerAgent()
            temp_agent.game_state = GameState()
            temp_agent.game_state.tableau.top_row = state.tableau.top_row[:]
            temp_agent.game_state.tableau.middle_row = state.tableau.middle_row[:]
            temp_agent.game_state.tableau.bottom_row = state.tableau.bottom_row[:]
            temp_agent.game_state.hand = state.hand[:]
            
            # Take the action and get reward
            next_state, reward, _, _ = temp_agent.step(action)
            
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        return best_action
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the value up the tree"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """Select child node using UCB1 formula with temperature"""
        log_parent_visits = math.log(node.visits)
        
        def ucb_value(child):
            if child.visits == 0:
                return float('inf')
            exploitation = child.value / child.visits
            exploration = self.exploration_weight * math.sqrt(log_parent_visits / child.visits)
            return exploitation + exploration
        
        # Add temperature to make selection more deterministic
        children = list(node.children.values())
        if not children:
            return None
            
        values = [ucb_value(child) for child in children]
        max_value = max(values)
        temperature = 0.1  # Lower temperature = more deterministic
        
        # Softmax selection
        exp_values = [math.exp((v - max_value) / temperature) for v in values]
        sum_exp = sum(exp_values)
        probs = [v / sum_exp for v in exp_values]
        
        return np.random.choice(children, p=probs)

def create_agent():
    """Create a new MCTS agent instance for each worker process"""
    return MCTSAgent(
        exploration_weight=0.5,
        max_simulations=100,
        max_depth=39
    )

def run_episode(episode_num: int) -> float:
    """Run a single episode and return the total reward"""
    # Create a new agent instance for this episode
    agent = create_agent()
    state = agent.agent.reset()
    episode_reward = 0
    done = False
    step_count = 0
    
    while not done:
        step_count += 1
        # Get action from MCTS
        action = agent.get_action(state)
        
        # Take the action
        next_state, reward, done, _ = agent.agent.step(action)
        
        # Update state and reward
        state = next_state
        episode_reward += reward
        
        # Early termination if we've taken too many steps
        if step_count > 100:  # Add a maximum step limit
            break
    
    return episode_reward

def run_episode_batch(batch_info: Tuple[int, int]) -> List[float]:
    """Run a batch of episodes and return their rewards"""
    batch_num, batch_size = batch_info
    rewards = []
    for _ in range(batch_size):
        reward = run_episode(batch_num * batch_size + _)  # Use actual episode number
        rewards.append(reward)
    return rewards

def train_mcts_agent(
    num_episodes: int = 1000,
    save_interval: int = 100,
    checkpoint_dir: str = "mcts_checkpoints",
    num_workers: int = None,
    batch_size: int = 10  # Process episodes in batches
) -> MCTSAgent:
    """Train the MCTS agent through self-play with parallel episodes"""
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Calculate number of batches
    num_batches = (num_episodes + batch_size - 1) // batch_size
    
    # Run episodes in parallel
    with mp.Pool(processes=num_workers) as pool:
        # Create batch information tuples (batch_num, batch_size)
        batch_info = [(i, batch_size) for i in range(num_batches)]
        
        # Use tqdm for progress tracking
        episode_rewards = []
        for batch_rewards in tqdm(
            pool.imap(run_episode_batch, batch_info),
            total=num_batches,
            desc="Training Episodes"
        ):
            episode_rewards.extend(batch_rewards)
            
            # Print progress
            if len(episode_rewards) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / 10
                print(f"\nLast 10 episodes average reward: {avg_reward:.2f}")
    
    # Create and return a new agent for testing
    return create_agent()

if __name__ == "__main__":
    # Train the agent
    agent = train_mcts_agent()
    
    # Example of using the trained agent
    state = agent.agent.reset()
    while not agent.agent.game_state.check_game_over():
        action = agent.get_action(state)
        state, reward, done, _ = agent.agent.step(action)
        print(f"Action: {action}, Reward: {reward}") 