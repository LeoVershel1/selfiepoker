import { Card, Tableau } from '../types';

const API_BASE_URL = 'http://localhost:5000/api';

export async function evaluateTableau(tableau: Tableau) {
    const response = await fetch(`${API_BASE_URL}/evaluate-tableau`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ tableau }),
    });

    if (!response.ok) {
        throw new Error('Failed to evaluate tableau');
    }

    return response.json();
}

export async function findOptimalArrangement(cards: Card[]) {
    const response = await fetch(`${API_BASE_URL}/find-optimal-arrangement`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ cards }),
    });

    if (!response.ok) {
        throw new Error('Failed to find optimal arrangement');
    }

    return response.json();
} 