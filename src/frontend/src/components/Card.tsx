import React from 'react';
import { Card as CardType, CardSource } from '../types';
import './Card.css';

interface CardProps {
    card: CardType;
    onDragStart?: (card: CardType) => void;
    source?: CardSource;
    sourceRowId?: string;
    isScoring?: boolean;
}

const CardComponent: React.FC<CardProps> = ({ card, onDragStart, source = 'hand', sourceRowId, isScoring = false }) => {
    const handleDragStart = (e: React.DragEvent<HTMLDivElement>) => {
        e.dataTransfer.setData('text/plain', JSON.stringify({ card, source, sourceRowId }));
        if (onDragStart) {
            onDragStart(card);
        }
    };

    const getSuitSymbol = (suit: string) => {
        // Handle both string and enum formats
        const suitLower = suit.toLowerCase();
        switch (suitLower) {
            case 'hearts':
            case '♥':
                return '♥';
            case 'diamonds':
            case '♦':
                return '♦';
            case 'clubs':
            case '♣':
                return '♣';
            case 'spades':
            case '♠':
                return '♠';
            default:
                return suit; // Return as is if it's already a symbol
        }
    };

    const getCardValue = (card: CardType) => {
        if (typeof card.value === 'number') {
            // Convert number to rank
            switch (card.value) {
                case 1: return 'A';
                case 11: return 'J';
                case 12: return 'Q';
                case 13: return 'K';
                default: return card.value.toString();
            }
        }
        return card.value;
    };

    const isRed = card.suit === '♥' || card.suit === '♦' || 
                 card.suit.toLowerCase() === 'hearts' || 
                 card.suit.toLowerCase() === 'diamonds';

    return (
        <div
            draggable
            onDragStart={handleDragStart}
            className={`card ${isRed ? 'red' : 'black'} ${isScoring ? 'scoring' : ''}`}
        >
            <div className="card-value">{getCardValue(card)}</div>
            <div className="card-suit">{getSuitSymbol(card.suit)}</div>
        </div>
    );
};

export default CardComponent; 