export type Suit = '♠' | '♥' | '♦' | '♣';
export type Value = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13;
export type CardSource = 'hand' | 'tableau';

export interface Card {
    rank: string;
    suit: Suit;
    value: Value;
    is_scoring: boolean;
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
    round: number;
    is_submitted: boolean;
}

export interface OptimalArrangementResult {
    arrangement: {
        top: Card[];
        middle: Card[];
        bottom: Card[];
    };
    scores: {
        immediate: number;
        future: number;
        total: number;
    };
    hand_types: {
        top: string;
        middle: string;
        bottom: string;
    };
}

export type CardDropHandler = (
    card: Card,
    rowId: string,
    position: number,
    source: CardSource,
    sourceRowId?: string
) => void; 