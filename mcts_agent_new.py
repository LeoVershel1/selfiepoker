import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
from agent import PokerAgent, GameObservation
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

class MCTSAgent:
    def __init__(
        self,
        exploration_weight: float = 1.0,
        max_simulations: int = 1000,
        max_depth: int = 39
    ):
        self.exploration_weight = exploration_weight
        self.max_simulations = max_simulations
        self.max_depth = max_depth
        self.agent = PokerAgent()
    
    def get_action(self, state: GameObservation) -> int:
        """Get the best action for the current state using MCTS"""
        print(f"\nDEBUG: Starting MCTS search with {len(state.hand)} cards")
        
        # Create root node
        root = MCTSNode(state)
        
        # Run simulations
        for i in range(self.max_simulations):
            print(f"\nDEBUG: Starting simulation {i+1}/{self.max_simulations}")
            
            # Selection
            node = self._select(root)
            print(f"DEBUG: Selected node with {len(node.children)} children")
            
            # Expansion
            if not node.state.tableau.is_complete():
                node = self._expand(node)
                print(f"DEBUG: Expanded node, now has {len(node.children)} children")
            
            # Simulation
            value = self._simulate(node)
            print(f"DEBUG: Simulation returned value: {value}")
            
            # Backpropagation
            self._backpropagate(node, value)
            print(f"DEBUG: Backpropagated value: {value}")
        
        # Choose best action
        if root.children:
            # Choose action with highest visit count
            best_action = max(root.children.items(), key=lambda x: x[1].visits)[0]
            best_value = root.children[best_action].value / max(root.children[best_action].visits, 1)
            print(f"DEBUG: Best action {best_action} with value {best_value}")
            return best_action
        
        print("DEBUG: No children in root node")
        return 0
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1"""
        print(f"DEBUG: Selecting from node with {len(node.children)} children")
        
        # If this is a leaf node or we have untried actions, expand it
        if not node.children or (node.untried_actions is not None and node.untried_actions):
            print("DEBUG: Found leaf node or untried actions, expanding")
            return self._expand(node)
        
        # Otherwise, traverse down the tree
        current = node
        while current.children and not current.state.tableau.is_complete():
            # If we have untried actions, expand
            if current.untried_actions and current.untried_actions:
                print("DEBUG: Found untried actions during traversal, expanding")
                return self._expand(current)
            
            # Select child using UCB1
            current = self._ucb_select(current)
            print(f"DEBUG: Selected child node with action {current.action}")
        
        return current
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
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
        # Copy tableau rows properly
        expand_agent.game_state.tableau.top_row = node.state.tableau.top_row[:]
        expand_agent.game_state.tableau.middle_row = node.state.tableau.middle_row[:]
        expand_agent.game_state.tableau.bottom_row = node.state.tableau.bottom_row[:]
        expand_agent.game_state.hand = node.state.hand[:]
        
        # Take the action
        next_state, reward, done, _ = expand_agent.step(action)
        print(f"DEBUG: Action {action} taken, reward: {reward}, done: {done}")
        
        # Create new node with the next state
        child = MCTSNode(next_state, parent=node, action=action)
        node.children[action] = child
        print(f"DEBUG: Created child node for action {action}")
        
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """Simulate a playout from the node using a simple strategy"""
        print(f"\nDEBUG: Starting simulation with {len(node.state.hand)} cards in hand")
        
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
                print("DEBUG: No valid actions available")
                break
            
            print(f"DEBUG: Depth {depth}, Valid actions: {valid_actions}")
            
            # Choose a random action
            action = np.random.choice(valid_actions)
            print(f"DEBUG: Random action chosen: {action}")
            
            # Take action
            next_state, reward, done, _ = sim_agent.step(action)
            print(f"DEBUG: Action {action} taken, reward: {reward}, done: {done}")
            
            # Check for invalid ordering
            is_invalid, penalty = sim_agent._check_invalid_ordering(next_state.tableau)
            if is_invalid:
                print(f"DEBUG: Invalid ordering detected, penalty: {penalty}")
                # Reduce the impact of invalid ordering penalty
                total_reward -= penalty * 0.5 * discount
                break
            
            # Calculate hand upgrade reward
            hand_upgrade_reward = sim_agent._get_hand_upgrade_reward(state.tableau, next_state.tableau)
            print(f"DEBUG: Hand upgrade reward: {hand_upgrade_reward}")
            
            # Update total reward with weighted components
            # Increase weight of hand improvements relative to direct rewards
            total_reward += (
                reward * 0.5 +  # Reduce impact of direct reward
                hand_upgrade_reward * 2.0  # Double the impact of hand improvements
            ) * discount
            
            print(f"DEBUG: Total reward so far: {total_reward}")
            
            # Update state for next iteration
            state = next_state
            discount *= 0.95  # Apply discount factor
            depth += 1
            
            if done:
                # Add terminal state reward
                if state.tableau.is_complete():
                    try:
                        score, _, is_game_over = sim_agent.game_state.evaluate_round()
                        if is_game_over:
                            print("DEBUG: Game over in terminal state")
                            total_reward -= 25 * discount  # Reduce game over penalty
                        else:
                            print(f"DEBUG: Round complete, score: {score}")
                            total_reward += score * 1.5 * discount  # Increase score reward
                    except ValueError:
                        print("DEBUG: Incomplete round in terminal state")
                        pass
                break
        
        print(f"DEBUG: Simulation complete, final reward: {total_reward}")
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
            if child.visits == 0:
                return float('inf')
            exploitation = child.value / child.visits
            exploration = self.exploration_weight * math.sqrt(log_parent_visits / child.visits)
            return exploitation + exploration
        
        return max(node.children.values(), key=ucb_value)

def train_mcts_agent(
    num_episodes: int = 1000,
    save_interval: int = 100,
    checkpoint_dir: str = "mcts_checkpoints"
) -> MCTSAgent:
    """Train the MCTS agent through self-play"""
    agent = MCTSAgent(
        exploration_weight=1.0,
        max_simulations=1000,
        max_depth=39
    )
    
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
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Average Reward: {episode_reward:.2f}")
    
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