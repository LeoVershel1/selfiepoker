from flask import Flask, request, jsonify
from flask_cors import CORS
from simple_game.poker import (
    evaluate_five_card_hand,
    evaluate_three_card_hand,
    calculate_five_card_score,
    calculate_three_card_score,
    HandType
)
from simple_game.card import Card as PyCard
from typing import List, Dict
import logging
from simple_game.solver import TableauSolver
from simple_game.card import Card, Suit

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def convert_to_pycard(card_dict: Dict) -> PyCard:
    """Convert a card dictionary from frontend to a Python Card object."""
    return PyCard(card_dict['value'], card_dict['suit'])

@app.route('/api/evaluate-row', methods=['POST'])
def evaluate_row():
    data = request.json
    logger.debug(f"Received row evaluation request: {data}")
    cards = [convert_to_pycard(card) for card in data['cards']]
    logger.debug(f"Converted cards: {cards}")

    # Determine which evaluation function to use
    if len(cards) == 3:
        hand_type, scoring_cards = evaluate_three_card_hand(cards)
        score = calculate_three_card_score(hand_type, scoring_cards)
    elif len(cards) == 5:
        # For API, assume not bottom row (middle row scoring)
        hand_type, scoring_cards = evaluate_five_card_hand(cards)
        score = calculate_five_card_score(hand_type, scoring_cards, False)
    else:
        return jsonify({'error': 'Row must have 3 or 5 cards'}), 400

    logger.debug(f"Hand type: {hand_type}, Score: {score}")

    response = {
        'score': score,
        'hand_type': str(hand_type),
        'scoring_cards': [{'suit': card.suit, 'value': card.value, 'id': f"{card.value}{card.suit}"} 
                         for card in scoring_cards]
    }
    logger.debug(f"Sending response: {response}")
    return jsonify(response)

@app.route('/api/evaluate-tableau', methods=['POST'])
def evaluate_tableau():
    data = request.json
    logger.debug(f"Received tableau evaluation request: {data}")
    tableau = data['tableau']
    row_scores = {}
    total_score = 0

    for row_id, row in tableau.items():
        logger.debug(f"Evaluating row {row_id}: {row}")
        cards = [convert_to_pycard(card) for card in row['cards']]
        if len(cards) == 3:
            hand_type, scoring_cards = evaluate_three_card_hand(cards)
            score = calculate_three_card_score(hand_type, scoring_cards)
        elif len(cards) == 5:
            # Determine if this is the bottom row for scoring
            is_bottom = row_id == 'bottom'
            hand_type, scoring_cards = evaluate_five_card_hand(cards)
            score = calculate_five_card_score(hand_type, scoring_cards, is_bottom)
        else:
            hand_type, scoring_cards, score = None, [], 0
        logger.debug(f"Row {row_id} - Hand type: {hand_type}, Score: {score}")
        row_scores[row_id] = {
            'score': score,
            'hand_type': str(hand_type) if hand_type else '',
            'scoring_cards': [{'suit': card.suit, 'value': card.value, 'id': f"{card.value}{card.suit}"} 
                            for card in scoring_cards]
        }
        total_score += score

    response = {
        'row_scores': row_scores,
        'total_score': total_score
    }
    logger.debug(f"Sending response: {response}")
    return jsonify(response)

@app.route('/api/find-optimal-arrangement', methods=['POST'])
def find_optimal_arrangement():
    data = request.json
    logger.debug(f"Received optimal arrangement request: {data}")
    
    try:
        # Convert cards to Python Card objects
        cards = [convert_to_pycard(card) for card in data['cards']]
        
        # Use solver to find best arrangement
        solver = TableauSolver()
        arrangement, immediate_score, future_value = solver.find_best_arrangement(cards)
        hand_types = solver.get_hand_types(arrangement)
        
        # Convert arrangement back to frontend format
        response = {
            'arrangement': {
                'top': [{'suit': card.suit, 'value': card.value, 'id': f"{card.value}{card.suit}"} 
                       for card in arrangement['top']],
                'middle': [{'suit': card.suit, 'value': card.value, 'id': f"{card.value}{card.suit}"} 
                          for card in arrangement['middle']],
                'bottom': [{'suit': card.suit, 'value': card.value, 'id': f"{card.value}{card.suit}"} 
                          for card in arrangement['bottom']]
            },
            'scores': {
                'immediate': immediate_score,
                'future': future_value,
                'total': (immediate_score * 0.7) + (future_value * 0.3)
            },
            'hand_types': {
                'top': str(hand_types['top']),
                'middle': str(hand_types['middle']),
                'bottom': str(hand_types['bottom'])
            }
        }
        
        logger.debug(f"Sending optimal arrangement response: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error finding optimal arrangement: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000) 