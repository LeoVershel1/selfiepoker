import React, { useState } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { Card as CardType, Tableau, GameState } from '../types';
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

    const handleCardDrop = (card: CardType, rowId: string, position: number) => {
        setGameState(prevState => {
            const newHand = prevState.hand.filter(c => c.id !== card.id);
            const newTableau = { ...prevState.tableau };
            newTableau[rowId as keyof Tableau].cards = [
                ...newTableau[rowId as keyof Tableau].cards.slice(0, position),
                card,
                ...newTableau[rowId as keyof Tableau].cards.slice(position)
            ];
            
            return {
                ...prevState,
                hand: newHand,
                tableau: newTableau
            };
        });
    };

    const handleSubmit = () => {
        // TODO: Implement scoring calculation
        console.log('Submitting tableau:', gameState.tableau);
    };

    return (
        <DndProvider backend={HTML5Backend}>
            <div className="game">
                <h1>Poker Hand Solver</h1>
                
                <div className="tableau">
                    <h2>Tableau</h2>
                    <TableauRow
                        row={gameState.tableau.bottom}
                        rowId="bottom"
                        onCardDrop={handleCardDrop}
                    />
                    <TableauRow
                        row={gameState.tableau.middle}
                        rowId="middle"
                        onCardDrop={handleCardDrop}
                    />
                    <TableauRow
                        row={gameState.tableau.top}
                        rowId="top"
                        onCardDrop={handleCardDrop}
                    />
                </div>

                <div className="hand">
                    <h2>Your Hand</h2>
                    <div className="hand-cards">
                        {gameState.hand.map(card => (
                            <div key={card.id} className="card-container">
                                <Card card={card} />
                            </div>
                        ))}
                    </div>
                </div>

                <button 
                    className="submit-button"
                    onClick={handleSubmit}
                    disabled={gameState.hand.length > 0}
                >
                    Submit Tableau
                </button>
            </div>
        </DndProvider>
    );
};

export default Game; 