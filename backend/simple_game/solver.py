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
from .cache_manager import SolverCache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Define suit order for consistent sorting
SUIT_ORDER = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.DIAMONDS: 2, Suit.CLUBS: 3}

def card_sort_key(card: Card) -> Tuple[int, int]:
    """Sort key for cards: (value, suit_order)"""
    return (CARD_VALUES[card.value], SUIT_ORDER[card.suit])

class TableauSolver:
    def __init__(self):
        self.best_score = 0
        self.best_arrangement = None
        self.best_future_value = 0
        self.cache = SolverCache()
    
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
        
        # Early pruning: Calculate immediate score first
        immediate_score = (
            calculate_three_card_score(top_type, arrangement['top']) +
            calculate_five_card_score(middle_type, arrangement['middle'], False) +
            calculate_five_card_score(bottom_type, arrangement['bottom'], True)
        )
        
        # If immediate score alone can't beat best_score, skip future value calculation
        if immediate_score * 0.85 <= self.best_score:
            return float('-inf'), 0, float('-inf')
        
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

    def _evaluate_arrangement_worker(self, arrangement_data: Tuple[Dict[str, List[Card]], List[Card]]) -> Optional[Tuple[Dict[str, List[Card]], int, float, float]]:
        """Worker function for parallel processing"""
        try:
            arrangement, cards = arrangement_data
            immediate_score, future_value, total_score = self.evaluate_arrangement(arrangement)
            if total_score != float('-inf'):
                return arrangement, immediate_score, future_value, total_score
        except Exception as e:
            print(f"Error in worker: {e}")
        return None

    def find_best_arrangement(self, cards: List[Card]) -> Tuple[Dict[str, List[Card]], int, float]:
        """Find the best possible arrangement by directly evaluating all valid arrangements"""
        if len(cards) != 13:
            raise ValueError("Must provide exactly 13 cards")

        # Check cache first
        cached_result = self.cache.get_cached_result(cards)
        if cached_result is not None:
            print("Found cached result!")
            return cached_result

        print("Finding best arrangement...")
        self.best_score = float('-inf')
        best_arrangement = None
        best_immediate_score = 0
        best_future_value = 0

        # Sort cards by value and suit to improve pruning efficiency
        sorted_cards = sorted(cards, key=card_sort_key, reverse=True)

        # Generate all possible arrangements
        arrangements = []
        for top_cards in combinations(sorted_cards, 3):
            # Early pruning: If top row is too weak, skip
            top_type, _ = evaluate_three_card_hand(top_cards)
            if top_type.value < HandType.HIGH_CARD.value:
                continue

            remaining_cards = [c for c in sorted_cards if c not in top_cards]
            
            for middle_cards in combinations(remaining_cards, 5):
                # Early pruning: If middle row is too weak, skip
                middle_type, _ = evaluate_five_card_hand(middle_cards)
                if middle_type.value <= top_type.value:
                    continue

                bottom_cards = [c for c in remaining_cards if c not in middle_cards]
                
                arrangement = {
                    'top': list(top_cards),
                    'middle': list(middle_cards),
                    'bottom': bottom_cards
                }
                arrangements.append((arrangement, cards))

        # Use parallel processing to evaluate arrangements
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU free
        print(f"Using {num_workers} workers for parallel processing")
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._evaluate_arrangement_worker, arr_data) 
                          for arr_data in arrangements]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        arrangement, immediate_score, future_value, total_score = result
                        if total_score > self.best_score:
                            self.best_score = total_score
                            best_arrangement = arrangement
                            best_immediate_score = immediate_score
                            best_future_value = future_value
                            print(f"Found better arrangement! Score: {self.best_score:.2f}")
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            # Fall back to sequential processing
            print("Falling back to sequential processing...")
            for arr_data in arrangements:
                result = self._evaluate_arrangement_worker(arr_data)
                if result is not None:
                    arrangement, immediate_score, future_value, total_score = result
                    if total_score > self.best_score:
                        self.best_score = total_score
                        best_arrangement = arrangement
                        best_immediate_score = immediate_score
                        best_future_value = future_value
                        print(f"Found better arrangement! Score: {self.best_score:.2f}")

        if best_arrangement is None or self.best_score == float('-inf'):
            raise ValueError("No valid arrangement found that satisfies row strength requirements")

        # Cache the result
        self.cache.cache_result(cards, best_arrangement, best_immediate_score, best_future_value)

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