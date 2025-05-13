import unittest
from game import GameState, PLACEMENT_SEQUENCE
from card import Card, Suit
from poker import (
    evaluate_five_card_hand, evaluate_three_card_hand,
    calculate_five_card_score, calculate_three_card_score,
    HandType
)

class TestSelfiePoker(unittest.TestCase):
    def setUp(self):
        self.game = GameState()

    def test_initial_game_state(self):
        """Test initial game setup"""
        # Check initial hand size
        self.assertEqual(len(self.game.hand), 6)
        
        # Check tableau is empty
        self.assertEqual(len(self.game.tableau.top_row), 0)
        self.assertEqual(len(self.game.tableau.middle_row), 0)
        self.assertEqual(len(self.game.tableau.bottom_row), 0)
        
        # Check placement index starts at 0
        self.assertEqual(self.game.tableau.current_placement_index, 0)

    def test_card_placement_sequence(self):
        """Test that cards are placed in the correct sequence"""
        # Play all 13 cards
        for _ in range(13):
            self.game.play_card(0)  # Always play first card in hand
        
        # Verify tableau structure
        self.assertEqual(len(self.game.tableau.top_row), 3)
        self.assertEqual(len(self.game.tableau.middle_row), 5)
        self.assertEqual(len(self.game.tableau.bottom_row), 5)
        
        # Verify placement sequence was followed
        for i, (row, slot) in enumerate(PLACEMENT_SEQUENCE):
            if row == "top":
                self.assertLess(slot, len(self.game.tableau.top_row))
            elif row == "middle":
                self.assertLess(slot, len(self.game.tableau.middle_row))
            elif row == "bottom":
                self.assertLess(slot, len(self.game.tableau.bottom_row))

    def test_round_evaluation(self):
        """Test round evaluation and scoring"""
        # Create a specific tableau to test scoring
        # Bottom row: Full House (strongest)
        self.game.tableau.bottom_row = [
            Card('A', Suit.HEARTS),
            Card('A', Suit.SPADES),
            Card('A', Suit.CLUBS),
            Card('K', Suit.HEARTS),
            Card('K', Suit.SPADES)
        ]
        
        # Middle row: Flush (middle strength)
        self.game.tableau.middle_row = [
            Card('2', Suit.DIAMONDS),
            Card('4', Suit.DIAMONDS),
            Card('6', Suit.DIAMONDS),
            Card('8', Suit.DIAMONDS),
            Card('10', Suit.DIAMONDS)
        ]
        
        # Top row: Three of a Kind (weakest)
        self.game.tableau.top_row = [
            Card('Q', Suit.HEARTS),
            Card('Q', Suit.SPADES),
            Card('Q', Suit.CLUBS)
        ]
        
        self.game.tableau.current_placement_index = 13  # Mark as complete
        
        # Evaluate round
        score, scoring_cards = self.game.evaluate_round()
        
        # Verify scoring cards
        # Full House: 5 cards (3 Aces + 2 Kings)
        # Flush: 5 cards (all diamonds)
        # Three of a Kind: 3 cards (all Queens)
        self.assertEqual(len(scoring_cards), 13)  # 5 + 5 + 3 = 13

    def test_card_redistribution(self):
        """Test card redistribution between rounds"""
        # Set up a complete round with known cards
        # Bottom row: Flush (strongest)
        self.game.tableau.bottom_row = [
            Card('2', Suit.DIAMONDS),
            Card('4', Suit.DIAMONDS),
            Card('6', Suit.DIAMONDS),
            Card('8', Suit.DIAMONDS),
            Card('10', Suit.DIAMONDS)
        ]
        # Middle row: Two Pair (middle strength)
        self.game.tableau.middle_row = [
            Card('K', Suit.HEARTS),
            Card('K', Suit.SPADES),
            Card('Q', Suit.HEARTS),
            Card('Q', Suit.SPADES),
            Card('2', Suit.HEARTS)
        ]
        # Top row: High Card (weakest)
        self.game.tableau.top_row = [
            Card('A', Suit.HEARTS),
            Card('3', Suit.SPADES),
            Card('5', Suit.CLUBS)
        ]
        self.game.tableau.current_placement_index = 13
        
        # Store initial unused draw pile
        initial_unused = self.game.unused_draw_pile.copy()
        
        # Evaluate and prepare next round
        score, scoring_cards = self.game.evaluate_round()
        self.game.prepare_next_round(scoring_cards)
        
        # Verify non-scoring cards were used for initial tableau
        # Non-scoring cards are:
        # - 2♥ from middle row (kicker in two pair)
        # - 3♠ and 5♣ from top row (not the highest card)
        non_scoring_cards = [
            Card('2', Suit.HEARTS),  # From middle row
            Card('3', Suit.SPADES),  # From top row
            Card('5', Suit.CLUBS)    # From top row
        ]
        for card in non_scoring_cards:
            found = False
            for row in [self.game.tableau.top_row, self.game.tableau.middle_row, self.game.tableau.bottom_row]:
                if card in row:
                    found = True
                    break
            self.assertTrue(found, f"Non-scoring card {card} should be in the new tableau")
        
        # Verify new draw pile contains scoring cards and unused cards
        new_draw_pile = self.game.unused_draw_pile
        self.assertEqual(len(new_draw_pile), len(scoring_cards) + len(initial_unused))

    def test_game_over_condition(self):
        """Test game over condition when hands are not in correct order"""
        # Set up a tableau where middle row is stronger than bottom row
        # Bottom row: Three of a Kind (weaker)
        self.game.tableau.bottom_row = [
            Card('A', Suit.HEARTS),
            Card('A', Suit.SPADES),
            Card('A', Suit.CLUBS),
            Card('2', Suit.HEARTS),
            Card('3', Suit.HEARTS)
        ]
        # Middle row: Flush (stronger)
        self.game.tableau.middle_row = [
            Card('2', Suit.DIAMONDS),
            Card('4', Suit.DIAMONDS),
            Card('6', Suit.DIAMONDS),
            Card('8', Suit.DIAMONDS),
            Card('10', Suit.DIAMONDS)
        ]
        # Top row: Pair (weakest)
        self.game.tableau.top_row = [
            Card('K', Suit.HEARTS),
            Card('K', Suit.SPADES),
            Card('Q', Suit.HEARTS)
        ]
        self.game.tableau.current_placement_index = 13
        
        # Verify game over condition
        with self.assertRaises(ValueError) as context:
            self.game.check_game_over()
        self.assertIn("Bottom row (THREE_OF_A_KIND) must be stronger than middle row (FLUSH)", str(context.exception))

if __name__ == '__main__':
    unittest.main() 