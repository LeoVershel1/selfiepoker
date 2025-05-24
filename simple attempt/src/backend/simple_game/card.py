from enum import Enum

class Suit(Enum):
    HEARTS = 'h'
    DIAMONDS = 'd'
    CLUBS = 'c'
    SPADES = 's'

    def __lt__(self, other):
        if not isinstance(other, Suit):
            return NotImplemented
        suit_order = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
        return suit_order[self.value] < suit_order[other.value]

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
    
    def __lt__(self, other: 'Card') -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        # Only compare by value, suits don't matter for ranking
        value_order = {'A': 14, 'K': 13, 'Q': 12, 'J': 11}
        self_val = value_order.get(self.value, int(self.value))
        other_val = value_order.get(other.value, int(other.value))
        return self_val < other_val
    
    def __gt__(self, other: 'Card') -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return other < self
    
    def __le__(self, other: 'Card') -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self < other or self == other
    
    def __ge__(self, other: 'Card') -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self > other or self == other
    
    @classmethod
    def from_string(cls, card_str: str) -> 'Card':
        if len(card_str) != 2:
            raise ValueError("Card string must be 2 characters (value + suit)")
        value, suit = card_str[0], card_str[1]
        return cls(value, Suit(suit)) 