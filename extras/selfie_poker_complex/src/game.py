from typing import List, Dict, Tuple
import random
from dataclasses import dataclass
from card import Card, Suit
from poker import (
    evaluate_five_card_hand, evaluate_three_card_hand,
    calculate_five_card_score, calculate_three_card_score,
    HandType
)

# Fixed order for card placement
PLACEMENT_SEQUENCE = [
    ("bottom", 0), ("middle", 0), ("top", 0),    # First three slots
    ("bottom", 1), ("middle", 1), ("top", 1),    # Next three slots
    ("bottom", 2), ("middle", 2), ("top", 2),    # Next three slots
    ("bottom", 3), ("middle", 3),                # Next two slots
    ("bottom", 4), ("middle", 4)                 # Final two slots
]

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

@dataclass
class Tableau:
    bottom_row: List[Card]  # 5 cards
    middle_row: List[Card]  # 5 cards
    top_row: List[Card]     # 3 cards
    current_placement_index: int = 0
    
    def __init__(self):
        self.bottom_row = []
        self.middle_row = []
        self.top_row = []
        self.current_placement_index = 0
    
    def is_complete(self) -> bool:
        return (len(self.bottom_row) == 5 and 
                len(self.middle_row) == 5 and 
                len(self.top_row) == 3)
    
    def get_next_empty_slot(self) -> Tuple[str, int]:
        """Returns (row_name, index) of next empty slot based on fixed sequence"""
        if self.current_placement_index >= len(PLACEMENT_SEQUENCE):
            return None
        return PLACEMENT_SEQUENCE[self.current_placement_index]

class GameState:
    def __init__(self):
        self.deck = Deck()
        self.hand: List[Card] = []
        self.tableau = Tableau()
        self.score = 0
        self.unused_draw_pile: List[Card] = []  # Track cards from draw pile not in hand/tableau
        self.initialize_game()
    
    def initialize_game(self):
        # Deal initial 6 cards to hand
        self.hand = [self.deck.draw() for _ in range(6)]
        # Rest of deck becomes initial unused draw pile
        self.unused_draw_pile = self.deck.cards.copy()
        self.deck.cards = []
    
    def play_card(self, hand_index: int):
        """Play a card from hand to the next slot in the sequence"""
        if hand_index >= len(self.hand):
            raise ValueError("Invalid hand index")
        
        next_slot = self.tableau.get_next_empty_slot()
        if next_slot is None:
            raise ValueError("No empty slots available")
        
        row, slot_index = next_slot
        card = self.hand.pop(hand_index)
        
        if row == "top":
            self.tableau.top_row.insert(slot_index, card)
        elif row == "middle":
            self.tableau.middle_row.insert(slot_index, card)
        elif row == "bottom":
            self.tableau.bottom_row.insert(slot_index, card)
        
        self.tableau.current_placement_index += 1
        
        # Draw a new card if unused draw pile isn't empty
        if self.unused_draw_pile:
            self.hand.append(self.unused_draw_pile.pop())

    def evaluate_round(self) -> Tuple[int, List[Card], bool]:
        """
        Evaluate the current round and return (score, cards_to_remove, is_game_over)
        Returns game_over=True if hands are in invalid order
        """
        if not self.tableau.is_complete():
            raise ValueError("Cannot evaluate incomplete round")

        # Evaluate each row
        top_hand_type, top_scoring_cards = evaluate_three_card_hand(self.tableau.top_row)
        middle_hand_type, middle_scoring_cards = evaluate_five_card_hand(self.tableau.middle_row)
        bottom_hand_type, bottom_scoring_cards = evaluate_five_card_hand(self.tableau.bottom_row)

        # Check if hands are in correct order (bottom > middle > top)
        if bottom_hand_type.value <= middle_hand_type.value or middle_hand_type.value <= top_hand_type.value:
            return 0, [], True  # Game over, no score, no cards to remove

        # Calculate scores
        top_score = calculate_three_card_score(top_hand_type, top_scoring_cards)
        middle_score = calculate_five_card_score(middle_hand_type, middle_scoring_cards, False)
        bottom_score = calculate_five_card_score(bottom_hand_type, bottom_scoring_cards, True)

        total_score = top_score + middle_score + bottom_score
        self.score += total_score

        # Collect all scoring cards
        scoring_cards = top_scoring_cards + middle_scoring_cards + bottom_scoring_cards

        return total_score, scoring_cards, False

    def prepare_next_round(self, scoring_cards: List[Card]):
        """Prepare the next round by redistributing cards"""
        # Get all non-scoring cards from the tableau
        non_scoring_cards = []
        for row in [self.tableau.top_row, self.tableau.middle_row, self.tableau.bottom_row]:
            non_scoring_cards.extend([card for card in row if card not in scoring_cards])

        # Clear the tableau and reset placement index
        self.tableau = Tableau()

        # Deal ALL non-scoring cards to the tableau in the fixed sequence
        random.shuffle(non_scoring_cards)
        for i, card in enumerate(non_scoring_cards):
            if i >= len(PLACEMENT_SEQUENCE):
                break
            row, slot_index = PLACEMENT_SEQUENCE[i]
            if row == "top":
                self.tableau.top_row.insert(slot_index, card)
            elif row == "middle":
                self.tableau.middle_row.insert(slot_index, card)
            elif row == "bottom":
                self.tableau.bottom_row.insert(slot_index, card)
            self.tableau.current_placement_index += 1

        # Create new draw pile from:
        # 1. All scoring cards from the previous round
        # 2. All unused cards from the previous draw pile
        new_draw_pile = scoring_cards + self.unused_draw_pile
        random.shuffle(new_draw_pile)
        
        # Update unused draw pile for next round
        self.unused_draw_pile = new_draw_pile

    def check_game_over(self) -> bool:
        """Check if the game is over due to invalid hand ordering"""
        if not self.tableau.is_complete():
            return False

        # Check current hand ordering
        if len(self.tableau.bottom_row) > 0 and len(self.tableau.middle_row) > 0:
            bottom_type, _, _ = evaluate_five_card_hand(self.tableau.bottom_row)
            middle_type, _, _ = evaluate_five_card_hand(self.tableau.middle_row)
            if bottom_type.value <= middle_type.value:
                return True

        if len(self.tableau.middle_row) > 0 and len(self.tableau.top_row) > 0:
            middle_type, _, _ = evaluate_five_card_hand(self.tableau.middle_row)
            top_type, _, _ = evaluate_three_card_hand(self.tableau.top_row)
            if middle_type.value <= top_type.value:
                return True

        return False

def main():
    game = GameState()
    
    while not game.check_game_over():
        # TODO: Implement game loop logic
        # 1. Display current game state
        # 2. Get player input for which card to play (no need to specify position)
        # 3. Process the move
        # 4. Check for round completion
        # 5. If round is complete, evaluate and prepare next round
        pass

if __name__ == "__main__":
    main()
