export type Suit = 'hearts' | 'diamonds' | 'clubs' | 'spades';
export type Value = '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10' | 'J' | 'Q' | 'K' | 'A';
export type CardSource = 'hand' | 'tableau';

export interface Card {
    suit: Suit;
    value: Value;
    id: string;  // Unique identifier for drag and drop
}

export interface TableauRow {
    cards: Card[];
    maxCards: number;
}

export interface Tableau {
    [key: string]: TableauRow;
    top: TableauRow;
    middle: TableauRow;
    bottom: TableauRow;
}

export interface GameState {
    hand: Card[];
    tableau: Tableau;
    score: number;
    futureValue: number;
}

export type CardDropHandler = (
    card: Card,
    rowId: string,
    position: number,
    source: CardSource,
    sourceRowId?: string
) => void; 