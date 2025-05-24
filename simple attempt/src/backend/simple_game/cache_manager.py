import os
import json
from typing import Dict, List, Tuple, Optional
from .card import Card, Suit

class SolverCache:
    def __init__(self, cache_dir: str = 'results'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_hand_key(self, cards: List[Card]) -> str:
        """Generate a unique key for a hand of cards"""
        # Sort cards by value and suit to ensure consistent ordering
        sorted_cards = sorted(cards, key=lambda c: (c.value, c.suit))
        return ','.join(f"{c.value}{c.suit.value}" for c in sorted_cards)
    
    def _get_cache_path(self, hand_key: str) -> str:
        """Get the path to the cache file for a hand"""
        return os.path.join(self.cache_dir, f"{hand_key}.json")
    
    def get_cached_result(self, cards: List[Card]) -> Optional[Tuple[Dict[str, List[Card]], int, float]]:
        """Get cached result for a hand if it exists"""
        hand_key = self._get_hand_key(cards)
        cache_path = self._get_cache_path(hand_key)
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                
            # Convert the cached data back to the expected format
            arrangement = {
                'top': [Card(value=c['value'], suit=Suit(c['suit'])) for c in data['arrangement']['top']],
                'middle': [Card(value=c['value'], suit=Suit(c['suit'])) for c in data['arrangement']['middle']],
                'bottom': [Card(value=c['value'], suit=Suit(c['suit'])) for c in data['arrangement']['bottom']]
            }
            
            return arrangement, data['immediate_score'], data['future_value']
        except Exception as e:
            print(f"Error reading cache: {e}")
            return None
    
    def cache_result(self, cards: List[Card], arrangement: Dict[str, List[Card]], 
                    immediate_score: int, future_value: float) -> None:
        """Cache a solver result"""
        hand_key = self._get_hand_key(cards)
        cache_path = self._get_cache_path(hand_key)
        
        # Convert cards to serializable format
        arrangement_data = {
            'top': [{'value': c.value, 'suit': c.suit.value} for c in arrangement['top']],
            'middle': [{'value': c.value, 'suit': c.suit.value} for c in arrangement['middle']],
            'bottom': [{'value': c.value, 'suit': c.suit.value} for c in arrangement['bottom']]
        }
        
        data = {
            'arrangement': arrangement_data,
            'immediate_score': immediate_score,
            'future_value': future_value
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error writing to cache: {e}") 