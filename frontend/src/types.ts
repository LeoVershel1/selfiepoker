export type Suit = 'hearts' | 'diamonds' | 'clubs' | 'spades';
export type Value = '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10' | 'J' | 'Q' | 'K' | 'A';

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