from typing import List, Tuple, Dict, Set, Optional
from itertools import combinations
from dataclasses import dataclass
from .card import Card, Suit
from .poker import (
    evaluate_five_card_hand, evaluate_three_card_hand,
    calculate_five_card_score, calculate_three_card_score,
    HandType, CARD_VALUES
)
from functools import lru_cache

@dataclass
class GameState:
    """Represents the current state of the game"""
    hand: List[Card]
    tableau: Dict[str, List[Card]]  # 'top', 'middle', 'bottom' rows
    score: int
    future_value: float
    
    def __init__(self, hand: List[Card]):
        self.hand = hand
        self.tableau = {
            'top': [],
            'middle': [],
            'bottom': []
        }
        self.score = 0
        self.future_value = 0
    
    def is_complete(self) -> bool:
        return (len(self.tableau['top']) == 3 and 
                len(self.tableau['middle']) == 5 and 
                len(self.tableau['bottom']) == 5)
    
    def get_state_key(self) -> str:
        """Generate a unique key for the current state for memoization"""
        hand_str = ','.join(sorted(f"{c.value}{c.suit}" for c in self.hand))
        tableau_str = '|'.join(
            ','.join(sorted(f"{c.value}{c.suit}" for c in self.tableau[row]))
            for row in ['top', 'middle', 'bottom']
        )
        return f"{hand_str}|{tableau_str}"
    
    def get_available_actions(self) -> List[Tuple[str, int, int]]:
        """Returns list of (row, position, card_index) tuples for valid moves"""
        actions = []
        for row in ['top', 'middle', 'bottom']:
            max_cards = 3 if row == 'top' else 5
            if len(self.tableau[row]) < max_cards:
                for pos in range(len(self.tableau[row]) + 1):
                    for card_idx in range(len(self.hand)):
                        actions.append((row, pos, card_idx))
        return actions

@dataclass
class Action:
    """Represents a move in the game"""
    row: str
    position: int
    card_index: int
    
    def __str__(self) -> str:
        return f"Place card {self.card_index} in {row} row at position {self.position}"

class TableauSolver:
    def __init__(self):
        self.best_score = 0
        self.best_arrangement = None
        self.best_future_value = 0
    
    @lru_cache(maxsize=1024)
    def evaluate_card_value(self, card: Card) -> float:
        """Evaluate a card's potential value for future rounds"""
        value = CARD_VALUES[card.value]
        if value >= 11:
            return value * 1.5
        if 5 <= value <= 10:
            return value * 1.2
        return value

    def evaluate_future_value(self, cards: List[Card]) -> float:
        """Evaluate the potential value of cards for future rounds"""
        if not cards:
            return 0
            
        values = sorted([CARD_VALUES[c.value] for c in cards])
        straight_potential = 0
        for i in range(len(values) - 1):
            if values[i+1] - values[i] <= 2:
                straight_potential += 1

        suits = [c.suit for c in cards]
        flush_potential = max(suits.count(s) for s in set(suits))

        value_counts = {}
        for c in cards:
            value_counts[CARD_VALUES[c.value]] = value_counts.get(CARD_VALUES[c.value], 0) + 1
        pair_potential = sum(1 for count in value_counts.values() if count >= 2)

        base_value = sum(self.evaluate_card_value(c) for c in cards)
        return base_value + (straight_potential * 5) + (flush_potential * 3) + (pair_potential * 10)

    def evaluate_arrangement(self, arrangement: Dict[str, List[Card]]) -> Tuple[float, float, float]:
        """Evaluate an arrangement and return (immediate_score, future_value, total_score)"""
        # Evaluate each row
        top_type, top_scoring = evaluate_three_card_hand(arrangement['top'])
        middle_type, middle_scoring = evaluate_five_card_hand(arrangement['middle'])
        bottom_type, bottom_scoring = evaluate_five_card_hand(arrangement['bottom'])
        
        # Check if hands are in correct order
        if bottom_type.value <= middle_type.value or middle_type.value <= top_type.value:
            return float('-inf'), 0, float('-inf')
        
        # Calculate immediate score
        immediate_score = (
            calculate_three_card_score(top_type, arrangement['top']) +
            calculate_five_card_score(middle_type, arrangement['middle'], False) +
            calculate_five_card_score(bottom_type, arrangement['bottom'], True)
        )
        
        # Calculate future value
        scoring_cards = top_scoring + middle_scoring + bottom_scoring
        all_cards = (arrangement['top'] + 
                    arrangement['middle'] + 
                    arrangement['bottom'])
        non_scoring_cards = [c for c in all_cards if c not in scoring_cards]
        future_value = self.evaluate_future_value(non_scoring_cards)
        
        # Combined score
        total_score = (immediate_score * 0.85) + (future_value * 0.15)
        
        return immediate_score, future_value, total_score

    def find_best_arrangement(self, cards: List[Card]) -> Tuple[Dict[str, List[Card]], int, float]:
        """Find the best possible arrangement by directly evaluating all valid arrangements"""
        if len(cards) != 13:
            raise ValueError("Must provide exactly 13 cards")

        print("Finding best arrangement...")
        best_score = float('-inf')
        best_arrangement = None
        best_immediate_score = 0
        best_future_value = 0

        # Generate all possible combinations for top row (3 cards)
        for top_cards in combinations(cards, 3):
            remaining_cards = [c for c in cards if c not in top_cards]
            
            # Generate all possible combinations for middle row (5 cards)
            for middle_cards in combinations(remaining_cards, 5):
                bottom_cards = [c for c in remaining_cards if c not in middle_cards]
                
                # Create arrangement
                arrangement = {
                    'top': list(top_cards),
                    'middle': list(middle_cards),
                    'bottom': bottom_cards
                }
                
                # Evaluate arrangement
                immediate_score, future_value, total_score = self.evaluate_arrangement(arrangement)
                
                # Update best if better
                if total_score > best_score:
                    best_score = total_score
                    best_arrangement = arrangement
                    best_immediate_score = immediate_score
                    best_future_value = future_value
                    
                    # Print progress
                    print(f"Found better arrangement! Score: {best_score:.2f}")

        if best_arrangement is None:
            raise ValueError("No valid arrangement found for these cards")

        return best_arrangement, best_immediate_score, best_future_value

    def get_hand_types(self, arrangement: Dict[str, List[Card]]) -> Dict[str, HandType]:
        """Get the hand types for each row in the arrangement"""
        top_type, _ = evaluate_three_card_hand(arrangement['top'])
        middle_type, _ = evaluate_five_card_hand(arrangement['middle'])
        bottom_type, _ = evaluate_five_card_hand(arrangement['bottom'])
        
        return {
            'top': top_type,
            'middle': middle_type,
            'bottom': bottom_type
        } 