import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './components/Home';
import Game from './components/Game';
import Solver from './components/Solver';
import './App.css';

function App() {
    return (
        <Router>
            <div className="app">
                <nav className="nav">
                    <Link to="/" className="nav-link">Home</Link>
                    <Link to="/game" className="nav-link">Play Game</Link>
                    <Link to="/solver" className="nav-link">Solver</Link>
                </nav>
                
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/game" element={<Game />} />
                    <Route path="/solver" element={<Solver />} />
                </Routes>
            </div>
        </Router>
    );
}

export default App;
