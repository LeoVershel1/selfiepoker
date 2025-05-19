import { Card, TableauRow } from '../types';

interface ScoringResult {
    score: number;
    hand_type: string;
    scoring_cards: Card[];
}

interface TableauScoringResult {
    row_scores: { [key: string]: ScoringResult };
    total_score: number;
}

interface OptimalArrangementResult {
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

// Helper function to get numeric value of a card
const getCardValue = (value: string): number => {
    switch (value) {
        case 'A': return 14;
        case 'K': return 13;
        case 'Q': return 12;
        case 'J': return 11;
        default: return parseInt(value);
    }
};

// Helper function to check if cards form a straight
const isStraight = (cards: Card[]): boolean => {
    if (cards.length < 5) return false;
    const values = cards.map(card => card.value).sort((a, b) => a - b);
    for (let i = 1; i < values.length; i++) {
        if (values[i] !== values[i - 1] + 1) return false;
    }
    return true;
};

// Helper function to check if cards are all the same suit
const isFlush = (cards: Card[]): boolean => {
    if (cards.length < 5) return false;
    const suit = cards[0].suit;
    return cards.every(card => card.suit === suit);
};

// Helper function to get groups of cards with the same value
const getGroups = (cards: Card[]): Card[][] => {
    const groups: { [key: string]: Card[] } = {};
    cards.forEach(card => {
        if (!groups[card.value]) {
            groups[card.value] = [];
        }
        groups[card.value].push(card);
    });
    return Object.values(groups);
};

// Evaluate a row of cards using the backend API
export const evaluateRow = async (row: TableauRow): Promise<ScoringResult> => {
    try {
        console.log('Evaluating row:', row);
        const response = await fetch('http://localhost:5000/api/evaluate-row', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ cards: row.cards }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Failed to evaluate row:', errorText);
            throw new Error(`Failed to evaluate row: ${errorText}`);
        }

        const result = await response.json();
        console.log('Row evaluation result:', result);
        return result;
    } catch (error) {
        console.error('Error evaluating row:', error);
        return { score: 0, hand_type: '', scoring_cards: [] };
    }
};

// Evaluate the entire tableau using the backend API
export const evaluateTableau = async (tableau: { [key: string]: TableauRow }): Promise<TableauScoringResult> => {
    try {
        console.log('Evaluating tableau:', tableau);
        const response = await fetch('http://localhost:5000/api/evaluate-tableau', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ tableau }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Failed to evaluate tableau:', errorText);
            throw new Error(`Failed to evaluate tableau: ${errorText}`);
        }

        const result = await response.json();
        console.log('Tableau evaluation result:', result);
        return result;
    } catch (error) {
        console.error('Error evaluating tableau:', error);
        return { row_scores: {}, total_score: 0 };
    }
};

// Find the optimal arrangement for a set of cards
export const findOptimalArrangement = async (cards: Card[]): Promise<OptimalArrangementResult> => {
    try {
        console.log('Finding optimal arrangement for cards:', cards);
        const response = await fetch('http://localhost:5000/api/find-optimal-arrangement', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ cards }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Failed to find optimal arrangement:', errorText);
            throw new Error(`Failed to find optimal arrangement: ${errorText}`);
        }

        const result = await response.json();
        console.log('Optimal arrangement result:', result);
        return result;
    } catch (error) {
        console.error('Error finding optimal arrangement:', error);
        throw error;
    }
}; 