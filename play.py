from game import GameState, Card
from typing import List
import os

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_card(card: Card) -> str:
    """Convert a card to a string representation."""
    return f"{card.value}{card.suit.value}"

def display_hand(hand: List[Card]) -> str:
    """Display the player's hand with indices."""
    return " ".join(f"[{i}]{display_card(card)}" for i, card in enumerate(hand))

def display_tableau(game_state: GameState) -> str:
    """Display the current tableau state."""
    # Create empty slots for incomplete rows
    top_row = game_state.tableau.top_row + ['__'] * (3 - len(game_state.tableau.top_row))
    middle_row = game_state.tableau.middle_row + ['__'] * (5 - len(game_state.tableau.middle_row))
    bottom_row = game_state.tableau.bottom_row + ['__'] * (5 - len(game_state.tableau.bottom_row))
    
    # Convert cards to string representation
    top_str = " ".join(display_card(card) if isinstance(card, Card) else card for card in top_row)
    middle_str = " ".join(display_card(card) if isinstance(card, Card) else card for card in middle_row)
    bottom_str = " ".join(display_card(card) if isinstance(card, Card) else card for card in bottom_row)
    
    return f"""
    Top:    {top_str}
    Middle: {middle_str}
    Bottom: {bottom_str}
    """

def display_game_state(game_state: GameState):
    """Display the complete game state."""
    clear_screen()
    print("\n=== Selfie Poker ===")
    print(f"Score: {game_state.score}")
    print("\nTableau:")
    print(display_tableau(game_state))
    print("\nYour Hand:")
    print(display_hand(game_state.hand))
    print("\nCards remaining in draw pile:", len(game_state.unused_draw_pile))

def get_player_move() -> int:
    """Get the player's move (card index to play)."""
    while True:
        try:
            move = input("\nEnter the number of the card to play (or 'q' to quit): ")
            if move.lower() == 'q':
                return -1
            card_index = int(move)
            if card_index < 0:
                print("Please enter a valid card number.")
                continue
            return card_index
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")

def play_game():
    """Main game loop."""
    game_state = GameState()
    
    while True:
        display_game_state(game_state)
        
        # Get player's move
        card_index = get_player_move()
        if card_index == -1:
            print("\nThanks for playing!")
            break
            
        try:
            # Validate move
            if card_index >= len(game_state.hand):
                print("Invalid card number!")
                continue
                
            # Play the card
            game_state.play_card(card_index)
            
            # Check if round is complete
            if game_state.tableau.is_complete():
                try:
                    score, scoring_cards = game_state.evaluate_round()
                    print(f"\nRound complete! Score: {score}")
                    print("Scoring cards:", " ".join(display_card(card) for card in scoring_cards))
                    input("\nPress Enter to continue...")
                    game_state.prepare_next_round(scoring_cards)
                except ValueError as e:
                    print(f"\nGame Over! {str(e)}")
                    print(f"Final Score: {game_state.score}")
                    break
                    
        except Exception as e:
            print(f"Error: {str(e)}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    play_game() 