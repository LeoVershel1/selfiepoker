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
    
    def __eq__(self, other: 'Card') -> bool:
        if not isinstance(other, Card):
            return False
        return self.value == other.value and self.suit == other.suit
    
    def __hash__(self) -> int:
        return hash((self.value, self.suit))
    
    @classmethod
    def from_string(cls, card_str: str) -> 'Card':
        if len(card_str) != 2:
            raise ValueError("Card string must be 2 characters (value + suit)")
        value, suit = card_str[0], card_str[1]
        return cls(value, Suit(suit)) 