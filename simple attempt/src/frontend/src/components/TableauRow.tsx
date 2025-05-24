import React from 'react';
import { Card as CardType, TableauRow as TableauRowType, CardDropHandler, CardSource } from '../types';
import Card from './Card';
import './TableauRow.css';

interface TableauRowProps {
    row: TableauRowType;
    rowId: string;
    onCardDrop: CardDropHandler;
    scoringCards: CardType[];
    rowScore: number;
    handType: string;
    isScoringComplete: boolean;
}

const TableauRow: React.FC<TableauRowProps> = ({ row, rowId, onCardDrop, scoringCards, rowScore, handType, isScoringComplete }) => {
    const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        const data = e.dataTransfer.getData('text/plain');
        if (data) {
            try {
                const { card, source, sourceRowId } = JSON.parse(data);
                const dropIndex = row.cards.length;
                if (dropIndex < row.maxCards) {
                    onCardDrop(card, rowId, dropIndex, source as CardSource, sourceRowId);
                }
            } catch (error) {
                console.error('Error parsing dropped card data:', error);
            }
        }
    };

    const isCardScoring = (card: CardType) => {
        console.log(`Checking card ${card.rank}${card.suit} (ID: ${card.id}) against scoring cards:`, scoringCards);
        const isScoring = scoringCards.some(sc => {
            const matches = sc.id === card.id;
            console.log(`Comparing with scoring card ${sc.rank}${sc.suit} (ID: ${sc.id}): ${matches}`);
            return matches;
        });
        console.log(`Card ${card.rank}${card.suit} is scoring: ${isScoring}`);
        return isScoring;
    };

    return (
        <div className="tableau-row-container">
            <div className="row-info">
                <div className="row-score">Score: {rowScore}</div>
                {handType && <div className="hand-type">{handType}</div>}
            </div>
            <div
                className="tableau-row"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
            >
                {row.cards.map((card, index) => (
                    <Card 
                        key={card.id} 
                        card={card} 
                        source="tableau"
                        sourceRowId={rowId}
                        isScoring={isCardScoring(card)}
                    />
                ))}
                {Array(row.maxCards - row.cards.length).fill(null).map((_, index) => (
                    <div key={`empty-${index}`} className="empty-slot" />
                ))}
            </div>
        </div>
    );
};

export default TableauRow; 