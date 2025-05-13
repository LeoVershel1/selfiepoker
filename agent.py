from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from card import Card, Suit
from game import GameState, Tableau
from poker import HandType, evaluate_five_card_hand, evaluate_three_card_hand

@dataclass
class GameObservation:
    """Represents the current state of the game from the agent's perspective"""
    hand: List[Card]
    tableau: Tableau
    
    @staticmethod
    def sort_cards(cards: List[Card]) -> List[Card]:
        """Sort cards by suit (h,d,c,s) and then by rank (2-10,J,Q,K,A)"""
        return sorted(cards, key=lambda card: (card.suit.value, CARD_VALUES[card.value]))
    
    def __post_init__(self):
        """Sort the hand after initialization"""
        self.hand = self.sort_cards(self.hand)

class PokerAgent:
    def __init__(self):
        self.game_state = GameState()
        self.observation_space = None  # Will be defined based on state representation
        self.action_space = None      # Will be defined based on possible actions
    
    def get_observation(self) -> GameObservation:
        """Convert current game state into an observation"""
        return GameObservation(
            hand=self.game_state.hand.copy(),
            tableau=self.game_state.tableau
        )
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid hand indices that can be played"""
        return list(range(len(self.game_state.hand)))
    
    def step(self, action: int) -> Tuple[GameObservation, float, bool, dict]:
        """
        Take an action in the environment and return the next state, reward, done flag, and info
        
        Args:
            action: Index of card in hand to play
            
        Returns:
            observation: Next state observation
            reward: Reward received from the action
            done: Whether the episode is finished
            info: Additional information
        """
        try:
            # Store current score for reward calculation
            old_score = self.game_state.score
            
            # Play the card
            self.game_state.play_card(action)
            
            # Check if round is complete
            if self.game_state.tableau.is_complete():
                try:
                    # Try to evaluate the round
                    score, scoring_cards = self.game_state.evaluate_round()
                    self.game_state.prepare_next_round(scoring_cards)
                    reward = score
                except ValueError as e:
                    # If evaluation fails, game is over
                    reward = -100  # Penalty for invalid hand ordering
                    return self.get_observation(), reward, True, {"error": str(e)}
            else:
                reward = 0  # No immediate reward for incomplete rounds
            
            # Check if game is over
            done = self.game_state.check_game_over()
            
            return self.get_observation(), reward, done, {}
            
        except Exception as e:
            # Handle any errors during the step
            return self.get_observation(), -100, True, {"error": str(e)}
    
    def reset(self) -> GameObservation:
        """Reset the game state and return initial observation"""
        self.game_state = GameState()
        return self.get_observation()
    
    def get_state_representation(self) -> np.ndarray:
        """
        Convert the current game state into a numerical representation
        that can be used by learning algorithms
        """
        # TODO: Implement state representation
        # This should include:
        # - Hand cards (one-hot encoded)
        # - Tableau state
        pass
    
    def get_action_representation(self, action: int) -> np.ndarray:
        """
        Convert an action into a numerical representation
        that can be used by learning algorithms
        """
        # TODO: Implement action representation
        pass
    
    def get_reward(self, old_score: int, new_score: int) -> float:
        """
        Calculate the reward for a state transition
        """
        return new_score - old_score 