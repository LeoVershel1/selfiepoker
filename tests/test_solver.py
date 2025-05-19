import unittest
from ..src.backend.simple_game.solver import TableauSolver
from ..src.backend.simple_game.card import Card, Suit
from ..src.backend.simple_game.poker import HandType

class TestTableauSolver(unittest.TestCase):
    def setUp(self):
        self.solver = TableauSolver()

    def test_all_same_suit(self):
        # Test with all cards of the same suit (hearts)
        cards = [
            Card('A', Suit.HEARTS), Card('K', Suit.HEARTS), Card('Q', Suit.HEARTS),
            Card('J', Suit.HEARTS), Card('10', Suit.HEARTS), Card('9', Suit.HEARTS),
            Card('8', Suit.HEARTS), Card('7', Suit.HEARTS), Card('6', Suit.HEARTS),
            Card('5', Suit.HEARTS), Card('4', Suit.HEARTS), Card('3', Suit.HEARTS),
            Card('2', Suit.HEARTS)
        ]
        arrangement, immediate_score, future_value = self.solver.find_best_arrangement(cards)
        
        # Verify the arrangement
        self.assertEqual(len(arrangement['top']), 3)
        self.assertEqual(len(arrangement['middle']), 5)
        self.assertEqual(len(arrangement['bottom']), 5)
        
        # Check that rows are in correct order (bottom > middle > top)
        top_type, _ = self.solver.get_hand_types(arrangement)['top']
        middle_type, _ = self.solver.get_hand_types(arrangement)['middle']
        bottom_type, _ = self.solver.get_hand_types(arrangement)['bottom']
        
        self.assertLess(top_type.value, middle_type.value)
        self.assertLess(middle_type.value, bottom_type.value)

    def test_multiple_full_houses(self):
        # Test with multiple full houses possible
        cards = [
            Card('A', Suit.HEARTS), Card('A', Suit.DIAMONDS), Card('A', Suit.CLUBS),
            Card('K', Suit.HEARTS), Card('K', Suit.DIAMONDS), Card('K', Suit.CLUBS),
            Card('Q', Suit.HEARTS), Card('Q', Suit.DIAMONDS), Card('Q', Suit.CLUBS),
            Card('J', Suit.HEARTS), Card('J', Suit.DIAMONDS), Card('J', Suit.CLUBS),
            Card('10', Suit.HEARTS)
        ]
        arrangement, immediate_score, future_value = self.solver.find_best_arrangement(cards)
        
        # Verify the arrangement
        self.assertEqual(len(arrangement['top']), 3)
        self.assertEqual(len(arrangement['middle']), 5)
        self.assertEqual(len(arrangement['bottom']), 5)
        
        # Check that rows are in correct order
        top_type, _ = self.solver.get_hand_types(arrangement)['top']
        middle_type, _ = self.solver.get_hand_types(arrangement)['middle']
        bottom_type, _ = self.solver.get_hand_types(arrangement)['bottom']
        
        self.assertLess(top_type.value, middle_type.value)
        self.assertLess(middle_type.value, bottom_type.value)

    def test_straight_flush_possible(self):
        # Test with a possible straight flush
        cards = [
            Card('A', Suit.HEARTS), Card('K', Suit.HEARTS), Card('Q', Suit.HEARTS),
            Card('J', Suit.HEARTS), Card('10', Suit.HEARTS), Card('9', Suit.HEARTS),
            Card('8', Suit.HEARTS), Card('7', Suit.HEARTS), Card('6', Suit.HEARTS),
            Card('5', Suit.HEARTS), Card('4', Suit.HEARTS), Card('3', Suit.HEARTS),
            Card('2', Suit.HEARTS)
        ]
        arrangement, immediate_score, future_value = self.solver.find_best_arrangement(cards)
        
        # Verify the arrangement
        self.assertEqual(len(arrangement['top']), 3)
        self.assertEqual(len(arrangement['middle']), 5)
        self.assertEqual(len(arrangement['bottom']), 5)
        
        # Check that rows are in correct order
        top_type, _ = self.solver.get_hand_types(arrangement)['top']
        middle_type, _ = self.solver.get_hand_types(arrangement)['middle']
        bottom_type, _ = self.solver.get_hand_types(arrangement)['bottom']
        
        self.assertLess(top_type.value, middle_type.value)
        self.assertLess(middle_type.value, bottom_type.value)

    def test_mixed_hands(self):
        # Test with a mix of different hand types
        cards = [
            Card('A', Suit.HEARTS), Card('A', Suit.DIAMONDS), Card('A', Suit.CLUBS),
            Card('K', Suit.HEARTS), Card('K', Suit.DIAMONDS), Card('K', Suit.CLUBS),
            Card('Q', Suit.HEARTS), Card('Q', Suit.DIAMONDS), Card('Q', Suit.CLUBS),
            Card('J', Suit.HEARTS), Card('J', Suit.DIAMONDS), Card('J', Suit.CLUBS),
            Card('10', Suit.HEARTS)
        ]
        arrangement, immediate_score, future_value = self.solver.find_best_arrangement(cards)
        
        # Verify the arrangement
        self.assertEqual(len(arrangement['top']), 3)
        self.assertEqual(len(arrangement['middle']), 5)
        self.assertEqual(len(arrangement['bottom']), 5)
        
        # Check that rows are in correct order
        top_type, _ = self.solver.get_hand_types(arrangement)['top']
        middle_type, _ = self.solver.get_hand_types(arrangement)['middle']
        bottom_type, _ = self.solver.get_hand_types(arrangement)['bottom']
        
        self.assertLess(top_type.value, middle_type.value)
        self.assertLess(middle_type.value, bottom_type.value)

    def test_royal_flush_possible(self):
        # Test with a possible royal flush
        cards = [
            Card('A', Suit.HEARTS), Card('K', Suit.HEARTS), Card('Q', Suit.HEARTS),
            Card('J', Suit.HEARTS), Card('10', Suit.HEARTS), Card('9', Suit.HEARTS),
            Card('8', Suit.HEARTS), Card('7', Suit.HEARTS), Card('6', Suit.HEARTS),
            Card('5', Suit.HEARTS), Card('4', Suit.HEARTS), Card('3', Suit.HEARTS),
            Card('2', Suit.HEARTS)
        ]
        arrangement, immediate_score, future_value = self.solver.find_best_arrangement(cards)
        
        # Verify the arrangement
        self.assertEqual(len(arrangement['top']), 3)
        self.assertEqual(len(arrangement['middle']), 5)
        self.assertEqual(len(arrangement['bottom']), 5)
        
        # Check that rows are in correct order
        top_type, _ = self.solver.get_hand_types(arrangement)['top']
        middle_type, _ = self.solver.get_hand_types(arrangement)['middle']
        bottom_type, _ = self.solver.get_hand_types(arrangement)['bottom']
        
        self.assertLess(top_type.value, middle_type.value)
        self.assertLess(middle_type.value, bottom_type.value)

if __name__ == '__main__':
    unittest.main() 