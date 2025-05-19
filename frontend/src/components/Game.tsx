import React, { useState } from 'react';
import { Card, Suit, Value, GameState } from '../types';
import { evaluateTableau, findOptimalArrangement } from '../utils/api';
import CardComponent from './Card';
import TableauRow from './TableauRow';
import './Game.css';

const RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];

const generateDeck = (): Card[] => {
    const deck: Card[] = [];
    const suits: Suit[] = ['♠', '♥', '♦', '♣'];
    const values: Value[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    
    suits.forEach(suit => {
        values.forEach(value => {
            deck.push({
                rank: RANKS[value - 1],
                suit,
                value,
                is_scoring: true,
                id: `${RANKS[value - 1]}${suit}${Date.now()}`
            });
        });
    });
    
    return deck;
};

const shuffleDeck = (deck: Card[]): Card[] => {
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
            round: 1,
            is_submitted: false
        };
    });

    const [scoringResults, setScoringResults] = useState<{
        rowScores: { [key: string]: { score: number; scoringCards: Card[]; handType: string } };
        totalScore: number;
    } | null>(null);

    const [optimalArrangement, setOptimalArrangement] = useState<{
        arrangement: { [key: string]: Card[] };
        scores: { immediate: number; future: number; total: number };
        hand_types: { [key: string]: string };
        rowScores?: { [key: string]: { score: number; scoringCards: Card[]; handType: string } };
    } | null>(null);

    const [isScoringComplete, setIsScoringComplete] = useState(false);

    const handleCardDrop = (card: Card, rowId: string, position: number, source: 'hand' | 'tableau', sourceRowId?: string) => {
        setGameState(prevState => {
            // Check if the card is already in the target row
            const targetRow = prevState.tableau[rowId as keyof typeof prevState.tableau];
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
                const sourceRow = newTableau[sourceRowId as keyof typeof newTableau];
                const sourceIndex = sourceRow.cards.findIndex(c => c.id === card.id);
                if (sourceIndex !== -1) {
                    sourceRow.cards.splice(sourceIndex, 1);
                }
            }

            // Add to target row
            newTableau[rowId as keyof typeof newTableau] = {
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

            // Find optimal arrangement
            const allCards = [
                ...gameState.tableau.top.cards,
                ...gameState.tableau.middle.cards,
                ...gameState.tableau.bottom.cards
            ];
            const optimal = await findOptimalArrangement(allCards);
            
            // Evaluate the optimal arrangement to get row scores
            const optimalResults = await evaluateTableau({
                top: { cards: optimal.arrangement.top, maxCards: 3 },
                middle: { cards: optimal.arrangement.middle, maxCards: 5 },
                bottom: { cards: optimal.arrangement.bottom, maxCards: 5 }
            });
            
            setOptimalArrangement({
                ...optimal,
                rowScores: {
                    top: {
                        score: optimalResults.row_scores.top.score,
                        scoringCards: optimalResults.row_scores.top.scoring_cards,
                        handType: optimalResults.row_scores.top.hand_type
                    },
                    middle: {
                        score: optimalResults.row_scores.middle.score,
                        scoringCards: optimalResults.row_scores.middle.scoring_cards,
                        handType: optimalResults.row_scores.middle.hand_type
                    },
                    bottom: {
                        score: optimalResults.row_scores.bottom.score,
                        scoringCards: optimalResults.row_scores.bottom.scoring_cards,
                        handType: optimalResults.row_scores.bottom.hand_type
                    }
                }
            });
            
            setIsScoringComplete(true);
        } catch (error) {
            console.error('Error submitting tableau:', error);
        }
    };

    const handleNewHand = () => {
        setGameState(prevState => {
            // Start with an empty hand
            const newHand: Card[] = [];
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
                    const row = prevState.tableau[rowId as keyof typeof prevState.tableau];
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
        setOptimalArrangement(null);
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
                <h2>Your Tableau</h2>
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

            {optimalArrangement && (
                <div className="optimal-arrangement">
                    <h2>Optimal Arrangement</h2>
                    <div className="score-comparison">
                        <div>Your Score: {scoringResults?.totalScore || 0}</div>
                        <div>Optimal Score: {optimalArrangement.scores.immediate}</div>
                        <div>Potential Improvement: +{optimalArrangement.scores.immediate - (scoringResults?.totalScore || 0)}</div>
                    </div>
                    <TableauRow
                        row={{ cards: optimalArrangement.arrangement.top, maxCards: 3 }}
                        rowId="top"
                        onCardDrop={() => {}}
                        scoringCards={optimalArrangement.rowScores?.top?.scoringCards || []}
                        rowScore={optimalArrangement.rowScores?.top?.score || 0}
                        handType={optimalArrangement.hand_types.top}
                    />
                    <TableauRow
                        row={{ cards: optimalArrangement.arrangement.middle, maxCards: 5 }}
                        rowId="middle"
                        onCardDrop={() => {}}
                        scoringCards={optimalArrangement.rowScores?.middle?.scoringCards || []}
                        rowScore={optimalArrangement.rowScores?.middle?.score || 0}
                        handType={optimalArrangement.hand_types.middle}
                    />
                    <TableauRow
                        row={{ cards: optimalArrangement.arrangement.bottom, maxCards: 5 }}
                        rowId="bottom"
                        onCardDrop={() => {}}
                        scoringCards={optimalArrangement.rowScores?.bottom?.scoringCards || []}
                        rowScore={optimalArrangement.rowScores?.bottom?.score || 0}
                        handType={optimalArrangement.hand_types.bottom}
                    />
                </div>
            )}

            <div className="hand">
                <h2>Your Hand</h2>
                <div className="hand-cards">
                    {gameState.hand.map(card => (
                        <CardComponent 
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