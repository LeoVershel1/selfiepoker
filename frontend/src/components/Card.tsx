import React from 'react';
import { useDrag, DragSourceMonitor } from 'react-dnd';
import { Card as CardType } from '../types';
import './Card.css';

interface CardProps {
    card: CardType;
}

const Card: React.FC<CardProps> = ({ card }) => {
    const [{ isDragging }, drag] = useDrag(() => ({
        type: 'CARD',
        item: { id: card.id, card },
        collect: (monitor: DragSourceMonitor) => ({
            isDragging: !!monitor.isDragging(),
        }),
    }));

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
            ref={drag}
            className={`card ${isDragging ? 'dragging' : ''} ${isRed ? 'red' : 'black'}`}
            style={{ opacity: isDragging ? 0.5 : 1 }}
        >
            <div className="card-value">{card.value}</div>
            <div className="card-suit">{getSuitSymbol(card.suit)}</div>
        </div>
    );
};

export default Card; 