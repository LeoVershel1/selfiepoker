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