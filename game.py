from typing import List, Dict, Tuple
import random
from dataclasses import dataclass
from enum import Enum

class Suit(Enum):
    HEARTS = 'h'
    DIAMONDS = 'd'
    CLUBS = 'c'
    SPADES = 's'

class Card:
    def __init__(self, value: str, suit: Suit):
        self.value = value
        self.suit = suit
    
    def __str__(self) -> str:
        return f"{self.value}{self.suit.value}"
    
    @classmethod
    def from_string(cls, card_str: str) -> 'Card':
        if len(card_str) != 2:
            raise ValueError("Card string must be 2 characters (value + suit)")
        value, suit = card_str[0], card_str[1]
        return cls(value, Suit(suit))

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
    
    def __init__(self):
        self.bottom_row = []
        self.middle_row = []
        self.top_row = []
    
    def is_complete(self) -> bool:
        return (len(self.bottom_row) == 5 and 
                len(self.middle_row) == 5 and 
                len(self.top_row) == 3)
    
    def get_next_empty_slot(self) -> Tuple[str, int]:
        """Returns (row_name, index) of next empty slot"""
        if len(self.top_row) < 3:
            return ("top", len(self.top_row))
        if len(self.middle_row) < 5:
            return ("middle", len(self.middle_row))
        if len(self.bottom_row) < 5:
            return ("bottom", len(self.bottom_row))
        return None

class GameState:
    def __init__(self):
        self.deck = Deck()
        self.hand: List[Card] = []
        self.tableau = Tableau()
        self.score = 0
        self.initialize_game()
    
    def initialize_game(self):
        # Deal initial 6 cards to hand
        self.hand = [self.deck.draw() for _ in range(6)]
    
    def play_card(self, hand_index: int, row: str, slot_index: int):
        """Play a card from hand to the specified slot"""
        if hand_index >= len(self.hand):
            raise ValueError("Invalid hand index")
        
        card = self.hand.pop(hand_index)
        
        if row == "top":
            self.tableau.top_row.insert(slot_index, card)
        elif row == "middle":
            self.tableau.middle_row.insert(slot_index, card)
        elif row == "bottom":
            self.tableau.bottom_row.insert(slot_index, card)
        else:
            raise ValueError("Invalid row")
        
        # Draw a new card if deck isn't empty
        if self.deck.cards:
            self.hand.append(self.deck.draw())

def main():
    game = GameState()
    
    while not game.tableau.is_complete():
        # TODO: Implement game loop logic
        # 1. Display current game state
        # 2. Get player input for card placement
        # 3. Process the move
        # 4. Check for game over conditions
        pass

if __name__ == "__main__":
    main()
