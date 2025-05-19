from typing import List
from card import Card, Suit
from solver import TableauSolver

def parse_card(card_str: str) -> Card:
    """Parse a card string (e.g., 'Ah' for Ace of Hearts) into a Card object"""
    if len(card_str) != 2:
        raise ValueError(f"Invalid card format: {card_str}")
    value, suit = card_str[0].upper(), card_str[1].lower()
    
    # Validate suit
    if suit not in ['h', 'd', 'c', 's']:
        raise ValueError(f"Invalid suit: {suit}")
    
    # Validate value
    valid_values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    if value not in valid_values:
        raise ValueError(f"Invalid value: {value}")
    
    return Card(value, Suit(suit))

def get_cards_from_user() -> List[Card]:
    """Get 13 cards from user input"""
    cards = []
    print("Enter 13 cards in format 'value+suit' (e.g., 'Ah' for Ace of Hearts)")
    print("Valid values: 2-10, J, Q, K, A")
    print("Valid suits: h (hearts), d (diamonds), c (clubs), s (spades)")
    
    while len(cards) < 13:
        try:
            card_str = input(f"Enter card {len(cards) + 1}: ").strip()
            card = parse_card(card_str)
            if card in cards:
                print("This card has already been entered")
                continue
            cards.append(card)
        except ValueError as e:
            print(f"Error: {e}")
    
    return cards

def display_arrangement(arrangement: dict, hand_types: dict, immediate_score: int, future_value: float):
    """Display the arrangement and its details"""
    print("\nBest Arrangement Found:")
    print("=" * 50)
    
    # Display bottom row
    print("\nBottom Row (5 cards):")
    print(f"Hand Type: {hand_types['bottom']}")
    print("Cards:", " ".join(str(card) for card in arrangement['bottom']))
    
    # Display middle row
    print("\nMiddle Row (5 cards):")
    print(f"Hand Type: {hand_types['middle']}")
    print("Cards:", " ".join(str(card) for card in arrangement['middle']))
    
    # Display top row
    print("\nTop Row (3 cards):")
    print(f"Hand Type: {hand_types['top']}")
    print("Cards:", " ".join(str(card) for card in arrangement['top']))
    
    print("\nScores:")
    print(f"Immediate Score: {immediate_score}")
    print(f"Future Value: {future_value:.1f}")
    print(f"Combined Score: {(immediate_score * 0.7 + future_value * 0.3):.1f}")
    print("=" * 50)

def main():
    solver = TableauSolver()
    
    print("Welcome to the Poker Hand Solver!")
    print("This program will find the best possible arrangement for your 13 cards.")
    print("The solver considers both immediate score and future value of cards.")
    
    while True:
        try:
            # Get cards from user
            cards = get_cards_from_user()
            
            # Find best arrangement
            arrangement, immediate_score, future_value = solver.find_best_arrangement(cards)
            hand_types = solver.get_hand_types(arrangement)
            
            # Display results
            display_arrangement(arrangement, hand_types, immediate_score, future_value)
            
            # Ask if user wants to try another hand
            if input("\nTry another hand? (y/n): ").lower() != 'y':
                break
                
        except ValueError as e:
            print(f"Error: {e}")
            if input("Try again? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    main() 