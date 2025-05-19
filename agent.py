from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from card import Card, Suit
from game import GameState, Tableau, PLACEMENT_SEQUENCE
from poker import (
    HandType, evaluate_five_card_hand, evaluate_three_card_hand, 
    CARD_VALUES, calculate_five_card_score, calculate_three_card_score
)

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
        # Define observation space dimensions
        self.num_cards = 52  # Total number of cards in deck
        self.num_slots = len(PLACEMENT_SEQUENCE)  # Total number of slots in tableau
        # Each slot needs 52 bits for one-hot encoding of its card
        self.observation_space = self.num_cards + (self.num_slots * self.num_cards)
        self.action_space = 6  # Maximum number of cards in hand
        
        # Reward weights
        self.score_weight = 0.7
        self.hand_upgrade_weight = 0.2
        self.invalid_ordering_weight = 0.1
    
    def get_observation(self) -> GameObservation:
        """Convert current game state into an observation"""
        return GameObservation(
            hand=self.game_state.hand.copy(),
            tableau=self.game_state.tableau
        )
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid hand indices that can be played"""
        return list(range(len(self.game_state.hand)))
    
    def card_to_index(self, card: Card) -> int:
        """Convert a card to its index in the one-hot encoding"""
        # Calculate index based on suit and value
        suit_offset = {'h': 0, 'd': 13, 'c': 26, 's': 39}[card.suit.value]
        value_index = list(CARD_VALUES.keys()).index(card.value)
        return suit_offset + value_index
    
    def get_state_representation(self) -> np.ndarray:
        """
        Convert the current game state into a numerical representation
        that can be used by learning algorithms
        
        Returns:
            np.ndarray: State representation where:
            - First 52 elements are one-hot encoded cards in hand
            - Next (13 * 52) elements represent tableau slots, where each slot has 52 bits
              for one-hot encoding of its card (if any)
        """
        # Initialize state array
        state = np.zeros(self.observation_space, dtype=np.float32)
        
        # One-hot encode cards in hand
        for card in self.game_state.hand:
            state[self.card_to_index(card)] = 1
        
        # Encode tableau state
        tableau_start = self.num_cards
        for i, (row, slot_idx) in enumerate(PLACEMENT_SEQUENCE):
            slot_offset = tableau_start + (i * self.num_cards)
            
            # Get the card in this slot (if any)
            card = None
            if row == "top" and slot_idx < len(self.game_state.tableau.top_row):
                card = self.game_state.tableau.top_row[slot_idx]
            elif row == "middle" and slot_idx < len(self.game_state.tableau.middle_row):
                card = self.game_state.tableau.middle_row[slot_idx]
            elif row == "bottom" and slot_idx < len(self.game_state.tableau.bottom_row):
                card = self.game_state.tableau.bottom_row[slot_idx]
            
            # If there's a card in this slot, one-hot encode it
            if card is not None:
                state[slot_offset + self.card_to_index(card)] = 1
        
        return state
    
    def get_action_representation(self, action: int) -> np.ndarray:
        """
        Convert an action into a numerical representation
        that can be used by learning algorithms
        
        Args:
            action: Index of card in hand to play
            
        Returns:
            np.ndarray: One-hot encoded action
        """
        action_vec = np.zeros(self.action_space, dtype=np.float32)
        action_vec[action] = 1
        return action_vec
    
    def _evaluate_hand_strength(self, cards: List[Card]) -> Tuple[HandType, List[Card], float]:
        """
        Evaluate hand strength based on number of cards.
        For partial hands, we evaluate what we have so far and their potential.
        Returns (hand_type, best_cards, hand_value)
        """
        if not cards:
            return HandType.HIGH_CARD, [], 0.0
            
        # For partial hands, we can still detect pairs, three of a kind, etc.
        values = [CARD_VALUES[card.value] for card in cards]
        value_counts = {}
        for value in values:
            value_counts[value] = value_counts.get(value, 0) + 1
        
        # Sort cards by value for consistent evaluation
        sorted_cards = sorted(cards, key=lambda x: CARD_VALUES[x.value], reverse=True)
        
        # For 3-card hands, we can only have high card, pair, or three of a kind
        if len(cards) == 3:
            # Check for three of a kind
            if 3 in value_counts.values():
                three_value = [v for v, count in value_counts.items() if count == 3][0]
                three_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == three_value]
                return HandType.THREE_OF_A_KIND, three_cards, three_value * 3
            
            # Check for pair
            if 2 in value_counts.values():
                pair_value = [v for v, count in value_counts.items() if count == 2][0]
                pair_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == pair_value]
                kicker = [card for card in sorted_cards if CARD_VALUES[card.value] != pair_value][0]
                return HandType.ONE_PAIR, pair_cards + [kicker], pair_value * 2 + CARD_VALUES[kicker.value]
            
            # High card
            return HandType.HIGH_CARD, [sorted_cards[0]], CARD_VALUES[sorted_cards[0].value]
        
        # For 5-card hands (partial or complete)
        if len(cards) <= 5:
            # Check for four of a kind
            if 4 in value_counts.values():
                four_value = [v for v, count in value_counts.items() if count == 4][0]
                four_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == four_value]
                kicker = [card for card in sorted_cards if CARD_VALUES[card.value] != four_value][0]
                return HandType.FOUR_OF_A_KIND, four_cards + [kicker], four_value * 4 + CARD_VALUES[kicker.value]
            
            # Check for full house
            if 3 in value_counts.values() and 2 in value_counts.values():
                three_value = [v for v, count in value_counts.items() if count == 3][0]
                pair_value = [v for v, count in value_counts.items() if count == 2][0]
                three_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == three_value]
                pair_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == pair_value]
                return HandType.FULL_HOUSE, three_cards + pair_cards, three_value * 3 + pair_value * 2
            
            # Check for three of a kind
            if 3 in value_counts.values():
                three_value = [v for v, count in value_counts.items() if count == 3][0]
                three_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == three_value]
                kickers = [card for card in sorted_cards if CARD_VALUES[card.value] != three_value][:2]
                return HandType.THREE_OF_A_KIND, three_cards + kickers, three_value * 3 + sum(CARD_VALUES[k.value] for k in kickers)
            
            # Check for two pair
            if list(value_counts.values()).count(2) == 2:
                pair_values = sorted([v for v, count in value_counts.items() if count == 2], reverse=True)
                pair_cards = []
                for value in pair_values:
                    pair_cards.extend([card for card in sorted_cards if CARD_VALUES[card.value] == value])
                kicker = [card for card in sorted_cards if CARD_VALUES[card.value] not in pair_values][0]
                return HandType.TWO_PAIR, pair_cards + [kicker], pair_values[0] * 2 + pair_values[1] * 2 + CARD_VALUES[kicker.value]
            
            # Check for one pair
            if 2 in value_counts.values():
                pair_value = [v for v, count in value_counts.items() if count == 2][0]
                pair_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == pair_value]
                kickers = [card for card in sorted_cards if CARD_VALUES[card.value] != pair_value][:3]
                return HandType.ONE_PAIR, pair_cards + kickers, pair_value * 2 + sum(CARD_VALUES[k.value] for k in kickers)
            
            # Check for potential flush
            # All placed cards must be of the same suit to consider it a potential flush
            suit_counts = {}
            for card in cards:
                suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
            
            if len(suit_counts) == 1:  # All cards are same suit
                return HandType.FLUSH, cards, sum(CARD_VALUES[card.value] for card in cards)
            
            # Check for potential straight
            # All placed cards must be part of a potential straight
            sorted_values = sorted(values)
            
            # Check for Ace-low straight (A-2-3-4-5)
            if set(sorted_values) == {14, 2, 3} or set(sorted_values) == {14, 2, 3, 4} or set(sorted_values) == {14, 2, 3, 4, 5}:
                return HandType.STRAIGHT, sorted_cards, 5  # Special value for Ace-low straight
            
            # Check for regular straight
            # For partial hands, all placed cards must be consecutive
            if len(sorted_values) >= 3:
                is_consecutive = True
                for i in range(len(sorted_values) - 1):
                    if sorted_values[i+1] - sorted_values[i] != 1:
                        is_consecutive = False
                        break
                if is_consecutive:
                    return HandType.STRAIGHT, sorted_cards, max(sorted_values)
            
            # High card
            return HandType.HIGH_CARD, [sorted_cards[0]], CARD_VALUES[sorted_cards[0].value]
        
        return HandType.HIGH_CARD, [sorted_cards[0]], CARD_VALUES[sorted_cards[0].value]
    
    def _check_invalid_ordering(self, tableau: Tableau) -> Tuple[bool, float]:
        """
        Check if current tableau state would lead to invalid ordering.
        Takes into account partial hands and their potential.
        Returns a tuple of (is_invalid, penalty_strength)
        """
        # Get the current hand types for each row
        bottom_type, _, _ = self._evaluate_hand_strength(tableau.bottom_row)
        middle_type, _, _ = self._evaluate_hand_strength(tableau.middle_row)
        top_type, _, _ = self._evaluate_hand_strength(tableau.top_row)
        
        # If any row is empty, we can't have invalid ordering yet
        if not tableau.bottom_row or not tableau.middle_row or not tableau.top_row:
            return False, 0.0
        
        # Check if current ordering is invalid
        if bottom_type.value <= middle_type.value or middle_type.value <= top_type.value:
            # Calculate penalty strength based on how bad the violation is
            if bottom_type.value <= middle_type.value:
                penalty = (middle_type.value - bottom_type.value + 1) * 5
            else:
                penalty = (top_type.value - middle_type.value + 1) * 5
            return True, penalty
        
        # For partial hands, we need to consider potential future improvements
        # For example, if bottom row has a pair and middle row has a pair,
        # but bottom row has higher cards, it might still be valid
        if len(tableau.bottom_row) == len(tableau.middle_row):
            # If same number of cards, compare the actual card values
            bottom_high = max(CARD_VALUES[card.value] for card in tableau.bottom_row)
            middle_high = max(CARD_VALUES[card.value] for card in tableau.middle_row)
            if bottom_high <= middle_high:
                return True, 3.0  # Penalty for having lower cards in bottom row
        
        if len(tableau.middle_row) == len(tableau.top_row):
            middle_high = max(CARD_VALUES[card.value] for card in tableau.middle_row)
            top_high = max(CARD_VALUES[card.value] for card in tableau.top_row)
            if middle_high <= top_high:
                return True, 3.0  # Penalty for having lower cards in middle row
        
        return False, 0.0
    
    def _get_hand_upgrade_reward(self, old_tableau: Tableau, new_tableau: Tableau) -> float:
        """
        Calculate reward for hand upgrades in each row.
        Takes into account partial hands and their potential.
        Rewards are structured to encourage building towards stronger hands.
        """
        reward = 0
        
        # Check bottom row
        if len(new_tableau.bottom_row) > len(old_tableau.bottom_row):
            old_type, _, old_value = self._evaluate_hand_strength(old_tableau.bottom_row)
            new_type, _, new_value = self._evaluate_hand_strength(new_tableau.bottom_row)
            
            # Major upgrade (e.g., high card -> pair -> three of a kind)
            if new_type.value > old_type.value:
                # Higher rewards for better hands, aligned with poker hand rankings
                if new_type == HandType.FOUR_OF_A_KIND:
                    reward += 10 + (new_value / 100)  # Add small bonus for higher card values
                elif new_type == HandType.FULL_HOUSE:
                    reward += 9 + (new_value / 100)
                elif new_type == HandType.FLUSH:
                    reward += 8 + (new_value / 100)
                elif new_type == HandType.STRAIGHT:
                    reward += 7 + (new_value / 100)
                elif new_type == HandType.THREE_OF_A_KIND:
                    reward += 6 + (new_value / 100)
                elif new_type == HandType.TWO_PAIR:
                    reward += 5.5 + (new_value / 100)
                elif new_type == HandType.ONE_PAIR:
                    reward += 5 + (new_value / 100)
                else:
                    reward += 4 + (new_value / 100)
            # Same type but improved value
            elif new_type.value == old_type.value:
                value_diff = new_value - old_value
                if value_diff > 0:
                    reward += value_diff / 10  # Reward proportional to value improvement
        
        # Similar logic for middle row (with lower rewards)
        if len(new_tableau.middle_row) > len(old_tableau.middle_row):
            old_type, _, old_value = self._evaluate_hand_strength(old_tableau.middle_row)
            new_type, _, new_value = self._evaluate_hand_strength(new_tableau.middle_row)
            
            if new_type.value > old_type.value:
                if new_type == HandType.FOUR_OF_A_KIND:
                    reward += 8 + (new_value / 100)
                elif new_type == HandType.FULL_HOUSE:
                    reward += 7 + (new_value / 100)
                elif new_type == HandType.FLUSH:
                    reward += 6 + (new_value / 100)
                elif new_type == HandType.STRAIGHT:
                    reward += 5 + (new_value / 100)
                elif new_type == HandType.THREE_OF_A_KIND:
                    reward += 4 + (new_value / 100)
                elif new_type == HandType.TWO_PAIR:
                    reward += 3.5 + (new_value / 100)
                elif new_type == HandType.ONE_PAIR:
                    reward += 3 + (new_value / 100)
                else:
                    reward += 2 + (new_value / 100)
            elif new_type.value == old_type.value:
                value_diff = new_value - old_value
                if value_diff > 0:
                    reward += value_diff / 15
        
        # Similar logic for top row (with lowest rewards)
        if len(new_tableau.top_row) > len(old_tableau.top_row):
            old_type, _, old_value = self._evaluate_hand_strength(old_tableau.top_row)
            new_type, _, new_value = self._evaluate_hand_strength(new_tableau.top_row)
            
            if new_type.value > old_type.value:
                if new_type == HandType.FOUR_OF_A_KIND:
                    reward += 6 + (new_value / 100)
                elif new_type == HandType.FULL_HOUSE:
                    reward += 5 + (new_value / 100)
                elif new_type == HandType.FLUSH:
                    reward += 4 + (new_value / 100)
                elif new_type == HandType.STRAIGHT:
                    reward += 3 + (new_value / 100)
                elif new_type == HandType.THREE_OF_A_KIND:
                    reward += 2 + (new_value / 100)
                elif new_type == HandType.TWO_PAIR:
                    reward += 1.5 + (new_value / 100)
                elif new_type == HandType.ONE_PAIR:
                    reward += 1 + (new_value / 100)
                else:
                    reward += 0.5 + (new_value / 100)
            elif new_type.value == old_type.value:
                value_diff = new_value - old_value
                if value_diff > 0:
                    reward += value_diff / 20
        
        return reward
    
    def get_reward(self, old_state: GameObservation, new_state: GameObservation) -> float:
        """
        Calculate reward based on:
        1. Score difference (major component)
        2. Hand upgrades (minor component)
        3. Penalties for invalid ordering
        """
        reward = 0
        
        # Check for invalid hand ordering (game over)
        if self.game_state.check_game_over():
            return -100
        
        # Check for invalid partial hands
        is_invalid, penalty = self._check_invalid_ordering(new_state.tableau)
        if is_invalid:
            reward -= penalty * self.invalid_ordering_weight
        
        # Score difference (if round is complete)
        if new_state.tableau.is_complete():
            try:
                score, _ = self.game_state.evaluate_round()
                reward += score * self.score_weight
            except ValueError:
                pass
        
        # Hand upgrade rewards
        hand_upgrade_reward = self._get_hand_upgrade_reward(old_state.tableau, new_state.tableau)
        reward += hand_upgrade_reward * self.hand_upgrade_weight
        
        return reward
    
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
            # Store current state for reward calculation
            old_state = self.get_observation()
            
            # Play the card
            self.game_state.play_card(action)
            
            # Get new state
            new_state = self.get_observation()
            
            # Calculate reward
            reward = self.get_reward(old_state, new_state)
            
            # Check if round is complete
            if self.game_state.tableau.is_complete():
                try:
                    # Try to evaluate the round
                    score, scoring_cards = self.game_state.evaluate_round()
                    self.game_state.prepare_next_round(scoring_cards)
                except ValueError as e:
                    # If evaluation fails, game is over
                    return self.get_observation(), -100, True, {"error": str(e)}
            
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