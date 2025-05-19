import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
import pickle
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from agent import PokerAgent, GameObservation
from card import Card
from game import GameState

class MCTSNode:
    def __init__(self, state: GameObservation, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children: Dict[int, MCTSNode] = {}  # action -> node
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None  # Will be initialized when needed
    
    def to_dict(self):
        """Convert node to dictionary for serialization"""
        return {
            'state': self.state,
            'action': self.action,
            'children': {action: child.to_dict() for action, child in self.children.items()},
            'visits': self.visits,
            'value': self.value,
            'untried_actions': self.untried_actions
        }
    
    @classmethod
    def from_dict(cls, data, parent=None):
        """Create node from dictionary"""
        node = cls(data['state'], parent, data['action'])
        node.visits = data['visits']
        node.value = data['value']
        node.untried_actions = data['untried_actions']
        node.children = {
            action: cls.from_dict(child_data, node)
            for action, child_data in data['children'].items()
        }
        return node

class MCTSAgent:
    def __init__(
        self,
        exploration_weight: float = 1.41,  # sqrt(2) for UCB1
        max_simulations: int = 1000,
        max_depth: int = 50,
        num_parallel: int = None
    ):
        self.exploration_weight = exploration_weight
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.agent = PokerAgent()
        self.num_parallel = num_parallel or cpu_count()
        self.tree_cache = {}  # Cache for storing partial trees
    
    def get_action(self, state: GameObservation) -> int:
        """Get the best action for the current state using parallel MCTS"""
        # Create a pool of workers
        with Pool(processes=self.num_parallel) as pool:
            # Run parallel MCTS simulations
            simulation_results = pool.map(
                partial(self._run_single_simulation, state),
                range(self.max_simulations)
            )
        
        # Aggregate results
        action_stats = defaultdict(lambda: {'visits': 0, 'value': 0.0})
        for action, value in simulation_results:
            action_stats[action]['visits'] += 1
            action_stats[action]['value'] += value
        
        # Choose the action with the highest visit count
        return max(action_stats.items(), key=lambda x: x[1]['visits'])[0]
    
    def _run_single_simulation(self, state: GameObservation, _) -> Tuple[int, float]:
        """Run a single MCTS simulation and return the first action and its value"""
        # Create a new agent instance for this simulation
        sim_agent = PokerAgent()
        sim_agent.game_state = GameState()  # Initialize new GameState
        sim_agent.game_state.tableau = state.tableau
        sim_agent.game_state.hand = state.hand
        
        root = MCTSNode(state)
        node = self._select(root)
        if not node.state.tableau.is_complete():
            node = self._expand(node, sim_agent)
        value = self._simulate(node, sim_agent)
        self._backpropagate(node, value)
        
        # Return the first action taken and its value
        if root.children:
            best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
            return best_action, root.children[best_action].value / root.children[best_action].visits
        return 0, 0.0
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1"""
        while node.children and not node.state.tableau.is_complete():
            if not all(action in node.children for action in self.agent.get_valid_actions()):
                return self._expand(node)
            node = self._ucb_select(node)
        return node
    
    def _expand(self, node: MCTSNode, sim_agent: PokerAgent) -> MCTSNode:
        """Expand the node by trying an untried action"""
        if node.untried_actions is None:
            node.untried_actions = sim_agent.get_valid_actions()
        
        if not node.untried_actions:
            return node
        
        action = node.untried_actions.pop()
        
        # Create new state by taking the action
        sim_agent.game_state.tableau = node.state.tableau
        sim_agent.game_state.hand = node.state.hand
        next_state, reward, done, _ = sim_agent.step(action)
        
        # Create new node
        child = MCTSNode(next_state, parent=node, action=action)
        node.children[action] = child
        return child
    
    def _simulate(self, node: MCTSNode, sim_agent: PokerAgent) -> float:
        """Simulate a random playout from the node"""
        state = node.state
        depth = 0
        total_reward = 0
        
        while not state.tableau.is_complete() and depth < self.max_depth:
            valid_actions = sim_agent.get_valid_actions()
            if not valid_actions:
                break
            
            # Choose random action
            action = np.random.choice(valid_actions)
            
            # Take action
            sim_agent.game_state.tableau = state.tableau
            sim_agent.game_state.hand = state.hand
            next_state, reward, done, _ = sim_agent.step(action)
            
            total_reward += reward
            state = next_state
            depth += 1
            
            if done:
                break
        
        return total_reward
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the value up the tree"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """Select child node using UCB1 formula"""
        log_parent_visits = math.log(node.visits)
        
        def ucb_value(child):
            exploitation = child.value / child.visits
            exploration = self.exploration_weight * math.sqrt(log_parent_visits / child.visits)
            return exploitation + exploration
        
        return max(node.children.values(), key=ucb_value)
    
    def save_state(self, path: str):
        """Save the agent's state to a file"""
        state = {
            'exploration_weight': self.exploration_weight,
            'max_simulations': self.max_simulations,
            'max_depth': self.max_depth,
            'num_parallel': self.num_parallel,
            'tree_cache': self.tree_cache
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: str):
        """Load the agent's state from a file"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.exploration_weight = state['exploration_weight']
        self.max_simulations = state['max_simulations']
        self.max_depth = state['max_depth']
        self.num_parallel = state['num_parallel']
        self.tree_cache = state['tree_cache']

def train_mcts_agent(
    num_episodes: int = 1000,
    save_interval: int = 100,
    checkpoint_dir: str = "mcts_checkpoints"
) -> MCTSAgent:
    """Train the MCTS agent through self-play with checkpoints"""
    agent = MCTSAgent()
    total_rewards = []
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for episode in range(num_episodes):
        # Reset the game state and get initial observation
        state = agent.agent.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from MCTS
            action = agent.get_action(state)
            
            # Take the action
            next_state, reward, done, _ = agent.agent.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"mcts_agent_ep{episode+1}.pkl")
            agent.save_state(checkpoint_path)
            print(f"Saved checkpoint at episode {episode + 1}")
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    # Save final state
    final_path = os.path.join(checkpoint_dir, "mcts_agent_final.pkl")
    agent.save_state(final_path)
    return agent

def load_trained_agent(checkpoint_path: str) -> MCTSAgent:
    """Load a trained MCTS agent from a checkpoint"""
    agent = MCTSAgent()
    agent.load_state(checkpoint_path)
    return agent

if __name__ == "__main__":
    # Train the agent
    agent = train_mcts_agent()
    
    # Example of using the trained agent
    state = agent.agent.reset()
    while not agent.agent.game_state.check_game_over():
        action = agent.get_action(state)
        state, reward, done, _ = agent.agent.step(action)
        print(f"Action: {action}, Reward: {reward}") 