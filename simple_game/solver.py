from typing import List, Tuple, Dict, Set, Optional
from itertools import combinations
from dataclasses import dataclass
from card import Card
from poker import (
    evaluate_five_card_hand, evaluate_three_card_hand,
    calculate_five_card_score, calculate_three_card_score,
    HandType, CARD_VALUES
)

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
        return f"Place card {self.card_index} in {self.row} row at position {self.position}"

class TableauSolver:
    def __init__(self):
        self.best_score = 0
        self.best_arrangement = None
        self.best_future_value = 0
        self.valuable_cards: Set[Card] = set()
    
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

    def apply_action(self, state: GameState, action: Action) -> Tuple[GameState, float, bool]:
        """
        Apply an action to the current state and return (new_state, reward, is_terminal)
        """
        new_state = GameState(state.hand.copy())
        new_state.tableau = {k: v.copy() for k, v in state.tableau.items()}
        
        # Move card from hand to tableau
        card = new_state.hand.pop(action.card_index)
        new_state.tableau[action.row].insert(action.position, card)
        
        # Calculate reward
        if new_state.is_complete():
            # Evaluate the complete arrangement
            top_type, top_scoring = evaluate_three_card_hand(new_state.tableau['top'])
            middle_type, middle_scoring = evaluate_five_card_hand(new_state.tableau['middle'])
            bottom_type, bottom_scoring = evaluate_five_card_hand(new_state.tableau['bottom'])
            
            # Check if hands are in correct order
            if bottom_type.value <= middle_type.value or middle_type.value <= top_type.value:
                return new_state, -100, True  # Invalid arrangement
            
            # Calculate immediate score
            immediate_score = (
                calculate_three_card_score(top_type, new_state.tableau['top']) +
                calculate_five_card_score(middle_type, new_state.tableau['middle'], False) +
                calculate_five_card_score(bottom_type, new_state.tableau['bottom'], True)
            )
            
            # Calculate future value
            scoring_cards = top_scoring + middle_scoring + bottom_scoring
            all_cards = (new_state.tableau['top'] + 
                        new_state.tableau['middle'] + 
                        new_state.tableau['bottom'])
            non_scoring_cards = [c for c in all_cards if c not in scoring_cards]
            future_value = self.evaluate_future_value(non_scoring_cards)
            
            # Combined reward
            reward = (immediate_score * 0.7) + (future_value * 0.3)
            return new_state, reward, True
        else:
            # Intermediate reward based on potential
            potential_reward = self.evaluate_future_value(new_state.hand) * 0.1
            return new_state, potential_reward, False

    def find_best_arrangement(self, cards: List[Card]) -> Tuple[Dict[str, List[Card]], int, float]:
        """
        Find the best possible arrangement using state-action-reward framework
        """
        if len(cards) != 13:
            raise ValueError("Must provide exactly 13 cards")

        self.best_score = 0
        self.best_arrangement = None
        self.best_future_value = 0

        # Initialize state
        initial_state = GameState(cards)
        
        def search(state: GameState, depth: int = 0) -> Tuple[float, Optional[Dict[str, List[Card]]]]:
            if state.is_complete():
                # Evaluate complete arrangement
                top_type, top_scoring = evaluate_three_card_hand(state.tableau['top'])
                middle_type, middle_scoring = evaluate_five_card_hand(state.tableau['middle'])
                bottom_type, bottom_scoring = evaluate_five_card_hand(state.tableau['bottom'])
                
                if bottom_type.value <= middle_type.value or middle_type.value <= top_type.value:
                    return float('-inf'), None
                
                immediate_score = (
                    calculate_three_card_score(top_type, state.tableau['top']) +
                    calculate_five_card_score(middle_type, state.tableau['middle'], False) +
                    calculate_five_card_score(bottom_type, state.tableau['bottom'], True)
                )
                
                scoring_cards = top_scoring + middle_scoring + bottom_scoring
                all_cards = (state.tableau['top'] + 
                            state.tableau['middle'] + 
                            state.tableau['bottom'])
                non_scoring_cards = [c for c in all_cards if c not in scoring_cards]
                future_value = self.evaluate_future_value(non_scoring_cards)
                
                return (immediate_score * 0.7) + (future_value * 0.3), state.tableau.copy()
            
            best_value = float('-inf')
            best_arrangement = None
            
            # Sort actions by potential value to find good solutions faster
            actions = state.get_available_actions()
            action_values = []
            for action in actions:
                new_state, reward, is_terminal = self.apply_action(state, Action(*action))
                action_values.append((action, reward))
            
            # Sort actions by reward in descending order
            action_values.sort(key=lambda x: x[1], reverse=True)
            
            # Only explore top 5 actions at each level to speed up search
            for action, _ in action_values[:5]:
                new_state, reward, is_terminal = self.apply_action(state, Action(*action))
                if is_terminal:
                    if reward > best_value:
                        best_value = reward
                        best_arrangement = new_state.tableau.copy()
                else:
                    value, arrangement = search(new_state, depth + 1)
                    if value > best_value:
                        best_value = value
                        best_arrangement = arrangement
            
            return best_value, best_arrangement

        # Start search
        print("Finding best arrangement...")
        best_value, best_arrangement = search(initial_state)
        
        if best_arrangement is None:
            raise ValueError("No valid arrangement found for these cards")

        # Calculate final scores
        top_type, _ = evaluate_three_card_hand(best_arrangement['top'])
        middle_type, _ = evaluate_five_card_hand(best_arrangement['middle'])
        bottom_type, _ = evaluate_five_card_hand(best_arrangement['bottom'])
        
        immediate_score = (
            calculate_three_card_score(top_type, best_arrangement['top']) +
            calculate_five_card_score(middle_type, best_arrangement['middle'], False) +
            calculate_five_card_score(bottom_type, best_arrangement['bottom'], True)
        )

        return best_arrangement, immediate_score, best_value - (immediate_score * 0.7)

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