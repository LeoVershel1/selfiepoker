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
from poker import CARD_VALUES, HandType
import random

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
        exploration_weight: float = 1.0,
        max_simulations: int = 2000,
        max_depth: int = 39,
        num_parallel: int = None,
        discount_factor: float = 0.95
    ):
        self.exploration_weight = exploration_weight
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.agent = PokerAgent()
        self.num_parallel = num_parallel or cpu_count()
        self.tree_cache = {}
        self.discount_factor = discount_factor
        
        # Get reward weights from the agent
        self.score_weight = self.agent.score_weight
        self.hand_upgrade_weight = self.agent.hand_upgrade_weight
        self.invalid_ordering_weight = self.agent.invalid_ordering_weight
        
        # Initialize root node
        self.root = None
    
    def get_action(self, state: GameObservation) -> int:
        """Get the best action for the current state using parallel MCTS"""
        # Try to find existing root node in cache
        state_key = str(state.tableau) + str(state.hand)
        if state_key in self.tree_cache:
            self.root = self.tree_cache[state_key]
        else:
            self.root = MCTSNode(state)
        
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
        
        # Choose the action with the highest average value
        valid_actions = self.agent.get_valid_actions()
        if not valid_actions:
            return 0
            
        # Filter to only valid actions and calculate average values
        action_values = {
            action: stats['value'] / max(stats['visits'], 1)
            for action, stats in action_stats.items()
            if action in valid_actions
        }
        
        if not action_values:
            return valid_actions[0]
        
        # Store the subtree for the chosen action
        best_action = max(action_values.items(), key=lambda x: x[1])[0]
        if best_action in self.root.children:
            next_state = self.root.children[best_action].state
            next_state_key = str(next_state.tableau) + str(next_state.hand)
            self.tree_cache[next_state_key] = self.root.children[best_action]
        
        return best_action
    
    def _run_single_simulation(self, state: GameObservation, _) -> Tuple[int, float]:
        """Run a single MCTS simulation and return the first action and its value"""
        print("\nDEBUG: Starting new simulation")
        
        # Create a new agent instance for this simulation
        sim_agent = PokerAgent()
        sim_agent.game_state = GameState()
        sim_agent.game_state.tableau = state.tableau
        sim_agent.game_state.hand = state.hand
        
        root = MCTSNode(state)
        print(f"DEBUG: Created root node with {len(state.hand)} cards")
        
        # Selection phase
        node = self._select(root)
        print(f"DEBUG: Selected node with {len(node.children)} children")
        
        # Expansion phase
        if not node.state.tableau.is_complete():
            node = self._expand(node, sim_agent)
            print(f"DEBUG: Expanded node, now has {len(node.children)} children")
        
        # Simulation phase
        value = self._simulate(node, sim_agent)
        print(f"DEBUG: Simulation returned value: {value}")
        
        # Backpropagation phase
        self._backpropagate(node, value)
        print(f"DEBUG: Backpropagated value: {value}")
        
        # Return the best action and its value
        if root.children:
            # Choose action with highest visit count
            best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
            best_value = root.children[best_action].value / max(root.children[best_action].visits, 1)
            print(f"DEBUG: Best action {best_action} with value {best_value}")
            return best_action, best_value
        
        print("DEBUG: No children in root node")
        return 0, 0.0
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1"""
        print(f"DEBUG: Selecting from node with {len(node.children)} children")
        
        # If this is a leaf node or we have untried actions, expand it
        if not node.children or (node.untried_actions is not None and node.untried_actions):
            print("DEBUG: Found leaf node or untried actions, expanding")
            return self._expand(node, self.agent)
        
        # Otherwise, traverse down the tree
        current = node
        while current.children and not current.state.tableau.is_complete():
            # If we have untried actions, expand
            if current.untried_actions and current.untried_actions:
                print("DEBUG: Found untried actions during traversal, expanding")
                return self._expand(current, self.agent)
            
            # Select child using UCB1
            current = self._ucb_select(current)
            print(f"DEBUG: Selected child node with action {current.action}")
        
        return current
    
    def _expand(self, node: MCTSNode, sim_agent: PokerAgent) -> MCTSNode:
        """Expand the node by trying an untried action"""
        print(f"DEBUG: Expanding node with {len(node.state.hand)} cards")
        
        # Initialize untried actions if not already done
        if node.untried_actions is None:
            # Get valid actions - just the indices of cards in hand
            node.untried_actions = list(range(len(node.state.hand)))
            print(f"DEBUG: Initialized untried actions: {node.untried_actions}")
        
        if not node.untried_actions:
            print("DEBUG: No untried actions available")
            return node
        
        # Get the next untried action
        action = node.untried_actions.pop()
        print(f"DEBUG: Trying action {action}")
        
        # Create a new agent instance for this expansion
        expand_agent = PokerAgent()
        expand_agent.game_state = GameState()
        # Create new lists for tableau and hand instead of trying to copy
        expand_agent.game_state.tableau = [[card for card in row] for row in node.state.tableau]
        expand_agent.game_state.hand = [card for card in node.state.hand]
        
        # Take the action
        next_state, reward, done, _ = expand_agent.step(action)
        print(f"DEBUG: Action {action} taken, reward: {reward}, done: {done}")
        
        # Create new node with the next state
        child = MCTSNode(next_state, parent=node, action=action)
        node.children[action] = child
        print(f"DEBUG: Created child node for action {action}")
        
        return child
    
    def _simulate(self, node: MCTSNode, sim_agent: PokerAgent) -> float:
        """Simulate a playout from the node using a more balanced strategy"""
        print(f"\nDEBUG: Starting simulation with {len(node.state.hand)} cards in hand")
        
        # Create a new agent instance for this simulation
        sim_agent = PokerAgent()
        sim_agent.game_state = GameState()
        # Create new lists for tableau and hand instead of trying to copy
        sim_agent.game_state.tableau = [[card for card in row] for row in node.state.tableau]
        sim_agent.game_state.hand = [card for card in node.state.hand]
        
        state = node.state
        depth = 0
        total_reward = 0
        discount = 1.0
        
        while not state.tableau.is_complete() and depth < self.max_depth:
            # Get valid actions
            valid_actions = sim_agent.get_valid_actions()
            if not valid_actions:
                print("DEBUG: No valid actions available")
                break
            
            print(f"DEBUG: Depth {depth}, Valid actions: {valid_actions}")
            
            # Evaluate potential hand improvements for each action
            action_scores = []
            for action in valid_actions:
                # Create a copy of the current state
                temp_agent = PokerAgent()
                temp_agent.game_state = GameState()
                temp_agent.game_state.tableau = [[card for card in row] for row in sim_agent.game_state.tableau]
                temp_agent.game_state.hand = [card for card in sim_agent.game_state.hand]
                
                # Take the action
                next_state, reward, done, _ = temp_agent.step(action)
                
                # Check for invalid ordering first
                is_invalid, penalty = temp_agent._check_invalid_ordering(next_state.tableau)
                if is_invalid:
                    print(f"DEBUG: Action {action} leads to invalid state, penalty: {penalty}")
                    action_scores.append((action, -penalty))
                    continue
                
                # Calculate hand upgrade reward
                hand_upgrade_reward = temp_agent._get_hand_upgrade_reward(state.tableau, next_state.tableau)
                print(f"DEBUG: Action {action} hand upgrade reward: {hand_upgrade_reward}")
                
                # Calculate potential score
                potential_score = 0
                if next_state.tableau.is_complete():
                    try:
                        score, _, is_game_over = temp_agent.game_state.evaluate_round()
                        if not is_game_over:
                            potential_score = score
                    except ValueError:
                        pass
                
                # Combine rewards with weights
                action_score = (
                    hand_upgrade_reward * 2.0 +  # Double weight for hand upgrades
                    potential_score * 0.5 +      # Half weight for potential score
                    reward                       # Direct reward from step
                )
                action_scores.append((action, action_score))
            
            if not action_scores:
                print("DEBUG: No valid actions after evaluation")
                break
            
            # Sort actions by their potential hand improvement
            action_scores.sort(key=lambda x: x[1], reverse=True)
            print(f"DEBUG: Action scores: {action_scores}")
            
            # Choose action with some randomness for exploration
            if random.random() < 0.2:  # 20% chance for random action
                action = random.choice(valid_actions)
                print(f"DEBUG: Random action chosen: {action}")
            else:
                # Choose from top 3 actions with probability weighted by their scores
                top_actions = action_scores[:min(3, len(action_scores))]
                actions = [a[0] for a in top_actions]
                weights = [max(0.1, a[1] + 100) for a in top_actions]  # Add offset to ensure positive weights
                weights = [w / sum(weights) for w in weights]  # Normalize weights
                action = random.choices(actions, weights=weights, k=1)[0]
                print(f"DEBUG: Weighted action chosen: {action} from {actions} with weights {weights}")
            
            # Take action
            next_state, reward, done, _ = sim_agent.step(action)
            print(f"DEBUG: Action {action} taken, reward: {reward}, done: {done}")
            
            # Check for invalid ordering
            is_invalid, penalty = sim_agent._check_invalid_ordering(next_state.tableau)
            if is_invalid:
                print(f"DEBUG: Invalid ordering detected, penalty: {penalty}")
                total_reward -= penalty * discount
                break
            
            # Calculate hand upgrade reward
            hand_upgrade_reward = sim_agent._get_hand_upgrade_reward(state.tableau, next_state.tableau)
            print(f"DEBUG: Hand upgrade reward: {hand_upgrade_reward}")
            
            # Update total reward
            total_reward += (hand_upgrade_reward * 2.0 + reward) * discount
            print(f"DEBUG: Total reward so far: {total_reward}")
            
            # Update state for next iteration
            state = next_state
            discount *= self.discount_factor
            depth += 1
            
            if done:
                # Add terminal state reward
                if state.tableau.is_complete():
                    try:
                        score, _, is_game_over = sim_agent.game_state.evaluate_round()
                        if is_game_over:
                            print("DEBUG: Game over in terminal state")
                            total_reward -= 50 * discount
                        else:
                            print(f"DEBUG: Round complete, score: {score}")
                            total_reward += score * discount
                    except ValueError:
                        print("DEBUG: Incomplete round in terminal state")
                        pass
                break
        
        print(f"DEBUG: Simulation complete, final reward: {total_reward}")
        return total_reward
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the value up the tree with improved value handling"""
        while node is not None:
            node.visits += 1
            # Use a more stable update formula with momentum
            alpha = 0.1  # Learning rate
            node.value = (1 - alpha) * node.value + alpha * value
            node = node.parent
    
    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """Select child node using UCB1 formula with improved exploration"""
        log_parent_visits = math.log(node.visits)
        
        def ucb_value(child):
            if child.visits == 0:
                return float('inf')
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
    num_episodes: int = 2000,
    save_interval: int = 100,
    checkpoint_dir: str = "mcts_checkpoints"
) -> MCTSAgent:
    """Train the MCTS agent through self-play with checkpoints"""
    agent = MCTSAgent(
        exploration_weight=0.5,  # Reduced exploration weight for more exploitation
        max_simulations=3000,    # Increased simulations for better search
        max_depth=39,
        discount_factor=0.99     # Increased discount factor for better long-term planning
    )
    total_rewards = []
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for episode in range(num_episodes):
        print(f"\nDEBUG: Starting episode {episode + 1}")
        # Reset the game state and get initial observation
        state = agent.agent.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            step_count += 1
            print(f"\nDEBUG: Step {step_count}")
            # Get action from MCTS
            action = agent.get_action(state)
            print(f"DEBUG: MCTS chose action {action}")
            
            # Take the action
            next_state, reward, done, _ = agent.agent.step(action)
            print(f"DEBUG: Step reward: {reward}, done: {done}")
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            print(f"DEBUG: Episode reward so far: {episode_reward}")
        
        print(f"DEBUG: Episode {episode + 1} complete. Total steps: {step_count}, Final reward: {episode_reward}")
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