import React, { useState } from 'react';
import { Card as CardType, Tableau, GameState } from '../types';
import { evaluateTableau } from '../utils/scoring';
import TableauRow from './TableauRow';
import Card from './Card';
import './Game.css';

const generateDeck = (): CardType[] => {
    const suits: Array<'hearts' | 'diamonds' | 'clubs' | 'spades'> = ['hearts', 'diamonds', 'clubs', 'spades'];
    const values: Array<'2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10' | 'J' | 'Q' | 'K' | 'A'> = 
        ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
    
    const deck: CardType[] = [];
    suits.forEach(suit => {
        values.forEach(value => {
            deck.push({
                suit,
                value,
                id: `${value}${suit}`
            });
        });
    });
    
    return deck;
};

const shuffleDeck = (deck: CardType[]): CardType[] => {
    const shuffled = [...deck];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
};

const Game: React.FC = () => {
    const [gameState, setGameState] = useState<GameState>(() => {
        const deck = shuffleDeck(generateDeck());
        return {
            hand: deck.slice(0, 13),
            tableau: {
                top: { cards: [], maxCards: 3 },
                middle: { cards: [], maxCards: 5 },
                bottom: { cards: [], maxCards: 5 }
            },
            score: 0,
            futureValue: 0
        };
    });

    const [scoringResults, setScoringResults] = useState<{
        rowScores: { [key: string]: { score: number; scoringCards: CardType[]; handType: string } };
        totalScore: number;
    } | null>(null);

    const [isScoringComplete, setIsScoringComplete] = useState(false);

    const handleCardDrop = (card: CardType, rowId: string, position: number, source: 'hand' | 'tableau', sourceRowId?: string) => {
        setGameState(prevState => {
            // Check if the card is already in the target row
            const targetRow = prevState.tableau[rowId as keyof Tableau];
            if (targetRow.cards.some(c => c.id === card.id)) {
                return prevState; // Don't update state if card is already in the row
            }

            // Create new state objects
            const newHand = [...prevState.hand];
            const newTableau = { ...prevState.tableau };

            // Handle the source of the card
            if (source === 'hand') {
                // Remove from hand
                const handIndex = newHand.findIndex(c => c.id === card.id);
                if (handIndex !== -1) {
                    newHand.splice(handIndex, 1);
                }
            } else if (source === 'tableau' && sourceRowId) {
                // Remove from source tableau row
                const sourceRow = newTableau[sourceRowId as keyof Tableau];
                const sourceIndex = sourceRow.cards.findIndex(c => c.id === card.id);
                if (sourceIndex !== -1) {
                    sourceRow.cards.splice(sourceIndex, 1);
                }
            }

            // Add to target row
            newTableau[rowId as keyof Tableau] = {
                ...targetRow,
                cards: [
                    ...targetRow.cards.slice(0, position),
                    card,
                    ...targetRow.cards.slice(position)
                ]
            };
            
            return {
                ...prevState,
                hand: newHand,
                tableau: newTableau
            };
        });
    };

    const handleSubmit = async () => {
        try {
            // Evaluate the tableau
            const results = await evaluateTableau(gameState.tableau);
            setScoringResults({
                rowScores: {
                    top: {
                        score: results.row_scores.top.score,
                        scoringCards: results.row_scores.top.scoring_cards,
                        handType: results.row_scores.top.hand_type
                    },
                    middle: {
                        score: results.row_scores.middle.score,
                        scoringCards: results.row_scores.middle.scoring_cards,
                        handType: results.row_scores.middle.hand_type
                    },
                    bottom: {
                        score: results.row_scores.bottom.score,
                        scoringCards: results.row_scores.bottom.scoring_cards,
                        handType: results.row_scores.bottom.hand_type
                    }
                },
                totalScore: results.total_score
            });
            setIsScoringComplete(true);
        } catch (error) {
            console.error('Error submitting tableau:', error);
        }
    };

    const handleNewHand = () => {
        setGameState(prevState => {
            // Start with an empty hand
            const newHand: CardType[] = [];
            // Create a new empty tableau
            const newTableau = {
                top: { cards: [], maxCards: 3 },
                middle: { cards: [], maxCards: 5 },
                bottom: { cards: [], maxCards: 5 }
            };
            let totalScore = prevState.score;

            if (scoringResults) {
                totalScore += scoringResults.totalScore;

                // Collect all non-scoring cards from the tableau
                Object.entries(scoringResults.rowScores).forEach(([rowId, result]) => {
                    const row = prevState.tableau[rowId as keyof Tableau];
                    const nonScoringCards = row.cards.filter(
                        card => !result.scoringCards.some(sc => sc.id === card.id)
                    );
                    newHand.push(...nonScoringCards);
                });
            }

            // Deal new cards to fill hand to 13
            const deck = shuffleDeck(generateDeck());
            const cardsNeeded = 13 - newHand.length;
            if (cardsNeeded > 0) {
                newHand.push(...deck.slice(0, cardsNeeded));
            }

            return {
                ...prevState,
                hand: newHand,
                tableau: newTableau,
                score: totalScore
            };
        });
        setScoringResults(null);
        setIsScoringComplete(false);
    };

    return (
        <div className="game">
            <h1>Poker Hand Solver</h1>
            <div className="score-display">
                Total Score: {gameState.score}
                {scoringResults && (
                    <div className="current-round-score">
                        Current Round: +{scoringResults.totalScore}
                    </div>
                )}
            </div>
            
            <div className="tableau">
                <h2>Tableau</h2>
                <TableauRow
                    row={gameState.tableau.top}
                    rowId="top"
                    onCardDrop={handleCardDrop}
                    scoringCards={scoringResults?.rowScores?.top?.scoringCards || []}
                    rowScore={scoringResults?.rowScores?.top?.score || 0}
                    handType={scoringResults?.rowScores?.top?.handType || ''}
                />
                <TableauRow
                    row={gameState.tableau.middle}
                    rowId="middle"
                    onCardDrop={handleCardDrop}
                    scoringCards={scoringResults?.rowScores?.middle?.scoringCards || []}
                    rowScore={scoringResults?.rowScores?.middle?.score || 0}
                    handType={scoringResults?.rowScores?.middle?.handType || ''}
                />
                <TableauRow
                    row={gameState.tableau.bottom}
                    rowId="bottom"
                    onCardDrop={handleCardDrop}
                    scoringCards={scoringResults?.rowScores?.bottom?.scoringCards || []}
                    rowScore={scoringResults?.rowScores?.bottom?.score || 0}
                    handType={scoringResults?.rowScores?.bottom?.handType || ''}
                />
            </div>

            <div className="hand">
                <h2>Your Hand</h2>
                <div className="hand-cards">
                    {gameState.hand.map(card => (
                        <Card 
                            key={card.id} 
                            card={card} 
                            source="hand"
                            isScoring={false}
                        />
                    ))}
                </div>
            </div>

            <div className="game-controls">
                <button 
                    className="submit-button"
                    onClick={handleSubmit}
                    disabled={gameState.hand.length > 0 || isScoringComplete}
                >
                    Submit Tableau
                </button>
                {isScoringComplete && (
                    <button 
                        className="new-hand-button"
                        onClick={handleNewHand}
                    >
                        New Hand
                    </button>
                )}
            </div>
        </div>
    );
};

export default Game; 