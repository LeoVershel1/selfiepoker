from typing import List, Dict, Tuple
from card import Card
from enum import Enum

# Card value mapping for easier comparison
CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

class HandType(Enum):
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10

    def __str__(self):
        return self.name.replace('_', ' ').title()

def get_card_values(cards: List[Card]) -> List[int]:
    """Convert card values to integers for comparison"""
    return [CARD_VALUES[card.value] for card in cards]

def is_flush(cards: List[Card]) -> bool:
    """Check if all cards are of the same suit"""
    return len(set(card.suit for card in cards)) == 1

def is_straight(card_values: List[int]) -> bool:
    """Check if cards form a straight"""
    values = sorted(card_values)
    # Handle Ace-low straight (A-2-3-4-5)
    if set(values) == {14, 2, 3, 4, 5}:
        return True
    # Check for regular straight
    return values == list(range(min(values), max(values) + 1))

def evaluate_five_card_hand(cards: List[Card]) -> Tuple[HandType, List[Card]]:
    """Evaluate a 5-card poker hand and return (hand_type, scoring_cards)"""
    if len(cards) != 5:
        raise ValueError("Must provide exactly 5 cards")
    
    values = get_card_values(cards)
    value_counts = {}
    for value in values:
        value_counts[value] = value_counts.get(value, 0) + 1
    
    # Sort cards by value for consistent evaluation
    sorted_cards = sorted(cards, key=lambda x: CARD_VALUES[x.value], reverse=True)
    
    # Check for flush and straight
    flush = is_flush(cards)
    straight = is_straight(values)
    
    # Royal Flush
    if flush and straight and max(values) == 14 and min(values) == 10:
        return HandType.ROYAL_FLUSH, sorted_cards
    
    # Straight Flush
    if flush and straight:
        return HandType.STRAIGHT_FLUSH, sorted_cards
    
    # Four of a Kind
    if 4 in value_counts.values():
        four_value = [v for v, count in value_counts.items() if count == 4][0]
        four_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == four_value]
        return HandType.FOUR_OF_A_KIND, four_cards
    
    # Full House
    if sorted(value_counts.values()) == [2, 3]:
        three_value = [v for v, count in value_counts.items() if count == 3][0]
        pair_value = [v for v, count in value_counts.items() if count == 2][0]
        three_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == three_value]
        pair_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == pair_value]
        return HandType.FULL_HOUSE, three_cards + pair_cards
    
    # Flush
    if flush:
        return HandType.FLUSH, sorted_cards
    
    # Straight
    if straight:
        return HandType.STRAIGHT, sorted_cards
    
    # Three of a Kind
    if 3 in value_counts.values():
        three_value = [v for v, count in value_counts.items() if count == 3][0]
        three_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == three_value]
        return HandType.THREE_OF_A_KIND, three_cards
    
    # Two Pair
    if list(value_counts.values()).count(2) == 2:
        pairs = [v for v, count in value_counts.items() if count == 2]
        pairs.sort(reverse=True)
        pair_cards = []
        for pair_value in pairs:
            pair_cards.extend([card for card in sorted_cards if CARD_VALUES[card.value] == pair_value])
        return HandType.TWO_PAIR, pair_cards
    
    # One Pair
    if 2 in value_counts.values():
        pair_value = [v for v, count in value_counts.items() if count == 2][0]
        pair_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == pair_value]
        return HandType.ONE_PAIR, pair_cards
    
    # High Card
    return HandType.HIGH_CARD, [sorted_cards[0]]

def evaluate_three_card_hand(cards: List[Card]) -> Tuple[HandType, List[Card]]:
    """Evaluate a 3-card poker hand and return (hand_type, scoring_cards)"""
    if len(cards) != 3:
        raise ValueError("Must provide exactly 3 cards")
    
    values = get_card_values(cards)
    value_counts = {}
    for value in values:
        value_counts[value] = value_counts.get(value, 0) + 1
    
    sorted_cards = sorted(cards, key=lambda x: CARD_VALUES[x.value], reverse=True)
    
    # Three of a Kind
    if len(set(values)) == 1:
        return HandType.THREE_OF_A_KIND, sorted_cards
    
    # One Pair
    if 2 in value_counts.values():
        pair_value = [v for v, count in value_counts.items() if count == 2][0]
        pair_cards = [card for card in sorted_cards if CARD_VALUES[card.value] == pair_value]
        return HandType.ONE_PAIR, pair_cards
    
    # High Card
    return HandType.HIGH_CARD, [sorted_cards[0]]

def calculate_five_card_score(hand_type: HandType, cards: List[Card], is_bottom_row: bool) -> int:
    """Calculate score for a 5-card hand based on position"""
    base_scores = {
        HandType.HIGH_CARD: [0, 0],
        HandType.ONE_PAIR: [1, 2],
        HandType.TWO_PAIR: [3, 6],
        HandType.THREE_OF_A_KIND: [10, 20],
        HandType.STRAIGHT: [20, 40],
        HandType.FLUSH: [30, 60],
        HandType.FULL_HOUSE: [40, 80],
        HandType.FOUR_OF_A_KIND: [60, 120],
        HandType.STRAIGHT_FLUSH: [150, 300],
        HandType.ROYAL_FLUSH: [250, 0]  # Royal flush can't be in middle row
    }
    
    return base_scores[hand_type][0 if is_bottom_row else 1]

def calculate_three_card_score(hand_type: HandType, cards: List[Card]) -> int:
    """Calculate score for a 3-card hand"""
    if hand_type == HandType.HIGH_CARD:
        return 0
    
    if hand_type == HandType.ONE_PAIR:
        # Get the value of the pair
        pair_value = CARD_VALUES[cards[0].value]
        # Score based on pair value
        if pair_value <= 10:  # 2-10
            return pair_value
        elif pair_value == 11:  # Jacks
            return 12
        elif pair_value == 12:  # Queens
            return 15
        elif pair_value == 13:  # Kings
            return 20
        else:  # Aces
            return 25
    
    if hand_type == HandType.THREE_OF_A_KIND:
        # Get the value of the three of a kind
        value = CARD_VALUES[cards[0].value]
        if value <= 10:  # 2-10
            return 60
        elif value == 11:  # Jacks
            return 70
        elif value == 12:  # Queens
            return 80
        elif value == 13:  # Kings
            return 90
        else:  # Aces
            return 100
    
    return 0 