from card import Card, Suit
from solver import TableauSolver
from typing import List

def create_test_hand(cards_str: List[str]) -> List[Card]:
    """Create a hand from a list of card strings (e.g., ['Ah', 'Kd', 'Qc'])"""
    return [Card.from_string(card_str) for card_str in cards_str]

# Predefined test hands
TEST_HANDS = {
    "royal_flush_test": [
        "Ah", "Kh", "Qh", "Jh", "10h",  # Royal flush in hearts
        "As", "Ks", "Qs", "Js", "10s",  # Royal flush in spades
        "Ac", "Kc", "Qc"                # High cards in clubs
    ],
    "straight_flush_test": [
        "9h", "8h", "7h", "6h", "5h",   # Straight flush in hearts
        "9d", "8d", "7d", "6d", "5d",   # Straight flush in diamonds
        "Ac", "Kc", "Qc"                # High cards in clubs
    ],
    "mixed_hands_test": [
        "Ah", "Ad", "Ac", "As", "Kc",   # Four of a kind
        "Kh", "Kd", "Ks", "Qh", "Qd",   # Full house
        "Jc", "10c", "9c"               # Straight
    ],
    "low_value_test": [
        "2h", "3h", "4h", "5h", "6h",   # Low straight flush
        "2d", "3d", "4d", "5d", "6d",   # Low straight flush
        "2c", "3c", "4c"                # Low straight
    ]
}

def test_hand(hand_name: str, cards: List[Card]):
    """Test a specific hand and display results"""
    print(f"\nTesting {hand_name}:")
    print("=" * 50)
    print("Cards:", " ".join(str(card) for card in cards))
    
    solver = TableauSolver()
    try:
        arrangement, immediate_score, future_value = solver.find_best_arrangement(cards)
        hand_types = solver.get_hand_types(arrangement)
        
        print("\nBest Arrangement Found:")
        print("-" * 30)
        
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
        
    except ValueError as e:
        print(f"Error: {e}")
    
    print("=" * 50)

def get_manual_hand() -> List[Card]:
    """Get a hand from user input"""
    print("\nEnter 13 cards in format 'value+suit' (e.g., 'Ah' for Ace of Hearts)")
    print("Valid values: 2-10, J, Q, K, A")
    print("Valid suits: h (hearts), d (diamonds), c (clubs), s (spades)")
    
    cards = []
    while len(cards) < 13:
        try:
            card_str = input(f"Enter card {len(cards) + 1}: ").strip()
            card = Card.from_string(card_str)
            if card in cards:
                print("This card has already been entered")
                continue
            cards.append(card)
        except ValueError as e:
            print(f"Error: {e}")
    
    return cards

def main():
    print("Poker Hand Solver Test")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Run predefined test hands")
        print("2. Enter a custom hand")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\nAvailable test hands:")
            test_hands_list = list(TEST_HANDS.keys())
            for i, hand_name in enumerate(test_hands_list, 1):
                print(f"{i}. {hand_name}")
            
            while True:
                test_choice = input("\nEnter test hand number: ").strip()
                try:
                    test_index = int(test_choice) - 1
                    if 0 <= test_index < len(test_hands_list):
                        hand_name = test_hands_list[test_index]
                        cards = create_test_hand(TEST_HANDS[hand_name])
                        test_hand(hand_name, cards)
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(test_hands_list)}")
                except ValueError:
                    print("Please enter a valid number")
        
        elif choice == "2":
            cards = get_manual_hand()
            test_hand("Custom Hand", cards)
        
        elif choice == "3":
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 