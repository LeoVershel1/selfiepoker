from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
from .card import Card, Suit
from poker import (
    evaluate_five_card_hand, evaluate_three_card_hand,
    calculate_five_card_score, calculate_three_card_score,
    HandType
)

@dataclass
class Tableau:
    bottom_row: List[Card]  # 5 cards
    middle_row: List[Card]  # 5 cards
    top_row: List[Card]     # 3 cards
    
    def __init__(self):
        self.bottom_row = []
        self.middle_row = []
        self.top_row = []
    
    def is_complete(self) -> bool:
        return (len(self.bottom_row) == 5 and 
                len(self.middle_row) == 5 and 
                len(self.top_row) == 3)
    
    def can_add_card(self, row: str, position: int) -> bool:
        """Check if a card can be added to the specified position"""
        if row == "top" and len(self.top_row) < 3 and position <= len(self.top_row):
            return True
        if row in ["middle", "bottom"] and len(getattr(self, f"{row}_row")) < 5 and position <= len(getattr(self, f"{row}_row")):
            return True
        return False
    
    def add_card(self, card: Card, row: str, position: int):
        """Add a card to the specified position"""
        if not self.can_add_card(row, position):
            raise ValueError(f"Cannot add card to {row} row at position {position}")
        getattr(self, f"{row}_row").insert(position, card)
    
    def remove_card(self, row: str, position: int) -> Card:
        """Remove a card from the specified position"""
        row_cards = getattr(self, f"{row}_row")
        if position >= len(row_cards):
            raise ValueError(f"No card at position {position} in {row} row")
        return row_cards.pop(position)

class GameState:
    def __init__(self):
        self.deck = Deck()
        self.hand: List[Card] = []
        self.tableau = Tableau()
        self.score = 0
        self.initialize_game()
    
    def initialize_game(self):
        """Deal initial 13 cards to hand"""
        self.hand = [self.deck.draw() for _ in range(13)]
    
    def move_card(self, hand_index: int, target_row: str, target_position: int):
        """Move a card from hand to tableau"""
        if hand_index >= len(self.hand):
            raise ValueError("Invalid hand index")
        
        card = self.hand.pop(hand_index)
        self.tableau.add_card(card, target_row, target_position)
    
    def move_card_between_rows(self, from_row: str, from_pos: int, to_row: str, to_pos: int):
        """Move a card between rows in the tableau"""
        card = self.tableau.remove_card(from_row, from_pos)
        self.tableau.add_card(card, to_row, to_pos)
    
    def evaluate_round(self) -> Tuple[int, List[Card]]:
        """Evaluate the current round and return (score, cards_to_remove)"""
        if not self.tableau.is_complete():
            raise ValueError("Cannot evaluate incomplete round")

        # Evaluate each row
        top_hand_type, top_scoring_cards = evaluate_three_card_hand(self.tableau.top_row)
        middle_hand_type, middle_scoring_cards = evaluate_five_card_hand(self.tableau.middle_row)
        bottom_hand_type, bottom_scoring_cards = evaluate_five_card_hand(self.tableau.bottom_row)

        # Check if hands are in correct order (bottom > middle > top)
        if bottom_hand_type.value <= middle_hand_type.value or middle_hand_type.value <= top_hand_type.value:
            return 0, []  # No score, no cards to remove

        # Collect all scoring cards
        scoring_cards = top_scoring_cards + middle_scoring_cards + bottom_scoring_cards
        
        # Calculate scores using original scoring system
        top_score = calculate_three_card_score(top_hand_type, top_scoring_cards)
        middle_score = calculate_five_card_score(middle_hand_type, middle_scoring_cards, False)  # False for middle row
        bottom_score = calculate_five_card_score(bottom_hand_type, bottom_scoring_cards, True)   # True for bottom row
        
        total_score = top_score + middle_score + bottom_score
        self.score += total_score
        
        return total_score, scoring_cards

    def prepare_next_round(self, scoring_cards: List[Card]):
        """Prepare the next round by dealing new cards"""
        # Get all non-scoring cards from the tableau
        non_scoring_cards = []
        for row in [self.tableau.top_row, self.tableau.middle_row, self.tableau.bottom_row]:
            non_scoring_cards.extend([card for card in row if card not in scoring_cards])

        # Clear the tableau
        self.tableau = Tableau()

        # Deal new cards to make up to 13
        cards_needed = 13 - len(non_scoring_cards)
        new_cards = [self.deck.draw() for _ in range(cards_needed)]
        
        # Combine non-scoring cards with new cards for next hand
        self.hand = non_scoring_cards + new_cards

class Deck:
    def __init__(self):
        self.cards: List[Card] = []
        self.reset()
    
    def reset(self):
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.cards = [Card(v, s) for s in Suit for v in values]
        self.shuffle()
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def draw(self) -> Card:
        if not self.cards:
            raise ValueError("No cards left in deck")
        return self.cards.pop()

def main():
    game = GameState()
    
    while True:
        # TODO: Implement game loop logic
        # 1. Display current game state
        # 2. Get player input for card movements
        # 3. Process the moves
        # 4. Check for round completion
        # 5. If round is complete, evaluate and prepare next round
        pass

if __name__ == "__main__":
    main() 