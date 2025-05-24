import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';

const Home: React.FC = () => {
    const navigate = useNavigate();

    return (
        <div className="home">
            <div className="home-content">
                <h1>Selfie Poker Solver</h1>
                <p className="description">
                    Arrange your cards to create the best possible poker hands.
                    Place cards in three rows: top (3 cards), middle (5 cards), and bottom (5 cards).
                    Each row must form a valid poker hand, with the bottom row being the strongest.
                </p>
                <button 
                    className="play-button"
                    onClick={() => navigate('/game')}
                >
                    Play Game
                </button>
            </div>
        </div>
    );
};

export default Home; 