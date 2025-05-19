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

const Card: React.FC<CardProps> = ({ card, onDragStart, source = 'hand', sourceRowId, isScoring = false }) => {
    const handleDragStart = (e: React.DragEvent<HTMLDivElement>) => {
        e.dataTransfer.setData('text/plain', JSON.stringify({ card, source, sourceRowId }));
        if (onDragStart) {
            onDragStart(card);
        }
    };

    const getSuitSymbol = (suit: string) => {
        switch (suit) {
            case 'hearts': return '♥';
            case 'diamonds': return '♦';
            case 'clubs': return '♣';
            case 'spades': return '♠';
            default: return '';
        }
    };

    const isRed = card.suit === 'hearts' || card.suit === 'diamonds';

    return (
        <div
            draggable
            onDragStart={handleDragStart}
            className={`card ${isRed ? 'red' : 'black'} ${isScoring ? 'scoring' : ''}`}
        >
            <div className="card-value">{card.value}</div>
            <div className="card-suit">{getSuitSymbol(card.suit)}</div>
        </div>
    );
};

export default Card; 