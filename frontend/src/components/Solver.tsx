import React, { useState } from 'react';
import { Card, Suit, Value, OptimalArrangementResult } from '../types';
import { findOptimalArrangement } from '../utils/scoring';
import CardComponent from './Card';
import './Solver.css';

const SUITS: Suit[] = ['♠', '♥', '♦', '♣'];
const RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];

const Solver: React.FC = () => {
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [optimalArrangement, setOptimalArrangement] = useState<OptimalArrangementResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCardSelect = (rank: string, suit: Suit) => {
    if (selectedCards.length >= 13) return;
    
    // Check if card is already selected
    const isDuplicate = selectedCards.some(
      card => card.rank === rank && card.suit === suit
    );
    
    if (isDuplicate) return;
    
    const newCard: Card = {
      rank,
      suit,
      value: (RANKS.indexOf(rank) + 1) as Value,
      is_scoring: true,
      id: `${rank}${suit}${Date.now()}`
    };
    
    setSelectedCards([...selectedCards, newCard]);
  };

  const handleRemoveCard = (index: number) => {
    setSelectedCards(selectedCards.filter((_, i) => i !== index));
  };

  const handleSolve = async () => {
    if (selectedCards.length !== 13) {
      setError('Please select exactly 13 cards');
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const result = await findOptimalArrangement(selectedCards);
      setOptimalArrangement(result);
    } catch (err) {
      setError('Failed to find optimal arrangement');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedCards([]);
    setOptimalArrangement(null);
    setError(null);
  };

  return (
    <div className="solver-container">
      <h1>Poker Hand Solver</h1>
      
      <div className="card-selection">
        <h2>Select Your Cards</h2>
        <div className="card-grid">
          {RANKS.map(rank => (
            SUITS.map(suit => (
              <button
                key={`${rank}${suit}`}
                className={`card-button ${suit === '♥' || suit === '♦' ? 'red' : 'black'}`}
                onClick={() => handleCardSelect(rank, suit)}
                disabled={selectedCards.length >= 13}
              >
                {rank}{suit}
              </button>
            ))
          ))}
        </div>
      </div>

      <div className="selected-cards">
        <h2>Selected Cards ({selectedCards.length}/13)</h2>
        <div className="selected-cards-grid">
          {selectedCards.map((card, index) => (
            <CardComponent
              key={index}
              card={card}
              source="hand"
            />
          ))}
        </div>
      </div>

      <div className="solver-controls">
        <button onClick={handleSolve} disabled={selectedCards.length !== 13 || isLoading}>
          {isLoading ? 'Solving...' : 'Find Optimal Arrangement'}
        </button>
        <button onClick={handleClear}>Clear</button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {optimalArrangement && (
        <div className="optimal-solution">
          <h2>Optimal Arrangement</h2>
          <div className="arrangement-grid">
            <div className="arrangement-row">
              <h3>Top Row</h3>
              <div className="tableau-row">
                {optimalArrangement.arrangement.top.map((card, index) => (
                  <CardComponent
                    key={index}
                    card={card}
                    source="tableau"
                    sourceRowId="top"
                  />
                ))}
              </div>
            </div>
            <div className="arrangement-row">
              <h3>Middle Row</h3>
              <div className="tableau-row">
                {optimalArrangement.arrangement.middle.map((card, index) => (
                  <CardComponent
                    key={index}
                    card={card}
                    source="tableau"
                    sourceRowId="middle"
                  />
                ))}
              </div>
            </div>
            <div className="arrangement-row">
              <h3>Bottom Row</h3>
              <div className="tableau-row">
                {optimalArrangement.arrangement.bottom.map((card, index) => (
                  <CardComponent
                    key={index}
                    card={card}
                    source="tableau"
                    sourceRowId="bottom"
                  />
                ))}
              </div>
            </div>
          </div>
          <div className="score-details">
            <h3>Scores</h3>
            <p>Immediate Score: {optimalArrangement.scores.immediate}</p>
            <p>Future Score: {optimalArrangement.scores.future}</p>
            <p>Total Score: {optimalArrangement.scores.total}</p>
          </div>
          <div className="hand-types">
            <h3>Hand Types</h3>
            <p>Top Row: {optimalArrangement.hand_types.top}</p>
            <p>Middle Row: {optimalArrangement.hand_types.middle}</p>
            <p>Bottom Row: {optimalArrangement.hand_types.bottom}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Solver; 