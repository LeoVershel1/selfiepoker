import React from 'react';
import { useDrop, DropTargetMonitor } from 'react-dnd';
import { Card as CardType, TableauRow as TableauRowType } from '../types';
import Card from './Card';
import './TableauRow.css';

interface TableauRowProps {
    row: TableauRowType;
    rowId: string;
    onCardDrop: (card: CardType, rowId: string, position: number) => void;
}

const TableauRow: React.FC<TableauRowProps> = ({ row, rowId, onCardDrop }) => {
    const [{ isOver }, drop] = useDrop(() => ({
        accept: 'CARD',
        drop: (item: { card: CardType }, monitor: DropTargetMonitor) => {
            if (monitor.didDrop()) {
                const dropIndex = row.cards.length;
                if (dropIndex < row.maxCards) {
                    onCardDrop(item.card, rowId, dropIndex);
                }
            }
        },
        collect: (monitor: DropTargetMonitor) => ({
            isOver: !!monitor.isOver({ shallow: true }),
        }),
    }));

    return (
        <div
            ref={drop}
            className={`tableau-row ${isOver ? 'drop-target' : ''}`}
        >
            {row.cards.map((card, index) => (
                <Card key={card.id} card={card} />
            ))}
            {Array(row.maxCards - row.cards.length).fill(null).map((_, index) => (
                <div key={`empty-${index}`} className="empty-slot" />
            ))}
        </div>
    );
};

export default TableauRow; 