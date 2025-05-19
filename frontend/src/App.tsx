import React, { useState } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Paper,
  Button,
  TextField,
  Card,
  CardContent,
  Stack
} from '@mui/material';
import Grid from '@mui/material/Grid';
import './App.css';

interface GameState {
  playerHand: string[];
  tableau: string[];
  suggestedMove: string;
}

interface CardProps {
  value: string;
  isSelected?: boolean;
  onClick?: () => void;
}

const formatCardInput = (input: string): string => {
  const inputLower = input.toLowerCase().trim();
  
  // Map of card values
  const valueMap: { [key: string]: string } = {
    'ace': 'A',
    'king': 'K',
    'queen': 'Q',
    'jack': 'J',
    '10': '10',
    '9': '9',
    '8': '8',
    '7': '7',
    '6': '6',
    '5': '5',
    '4': '4',
    '3': '3',
    '2': '2'
  };

  // Map of suits
  const suitMap: { [key: string]: string } = {
    'hearts': '♥',
    'diamonds': '♦',
    'clubs': '♣',
    'spades': '♠'
  };

  // Try to match the input pattern
  for (const [value, valueSymbol] of Object.entries(valueMap)) {
    for (const [suit, suitSymbol] of Object.entries(suitMap)) {
      if (inputLower.includes(value) && inputLower.includes(suit)) {
        return `${valueSymbol}${suitSymbol}`;
      }
    }
  }

  return input; // Return original input if no match found
};

const generateRandomCard = (): string => {
  const values = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'];
  const suits = ['♥', '♦', '♣', '♠'];
  const randomValue = values[Math.floor(Math.random() * values.length)];
  const randomSuit = suits[Math.floor(Math.random() * suits.length)];
  return `${randomValue}${randomSuit}`;
};

const generateRandomHand = (): string[] => {
  const hand: string[] = [];
  for (let i = 0; i < 6; i++) {
    hand.push(generateRandomCard());
  }
  return hand;
};

const PlayingCard: React.FC<CardProps> = ({ value, isSelected, onClick }) => {
  const isRed = value.includes('♥') || value.includes('♦');
  
  return (
    <Card 
      sx={{ 
        width: 80, 
        height: 120, 
        cursor: 'pointer',
        border: isSelected ? '2px solid #1976d2' : '1px solid #ccc',
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-5px)',
          boxShadow: 3
        }
      }}
      onClick={onClick}
    >
      <CardContent sx={{ 
        p: 1, 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        bgcolor: value ? '#fff' : '#f5f5f5'
      }}>
        <Typography 
          variant="h6" 
          align="center"
          sx={{ 
            color: isRed ? '#d40000' : '#000',
            fontWeight: 'bold'
          }}
        >
          {value || ''}
        </Typography>
      </CardContent>
    </Card>
  );
};

const TableauRow: React.FC<{
  cards: string[];
  onCardClick: (index: number) => void;
  startIndex: number;
  count: number;
}> = ({ cards, onCardClick, startIndex, count }) => {
  return (
    <Stack direction="row" spacing={1} sx={{ justifyContent: 'center', mb: 2 }}>
      {Array.from({ length: count }).map((_, i) => {
        const index = startIndex + i;
        return (
          <PlayingCard
            key={index}
            value={cards[index] || ''}
            isSelected={!!cards[index]}
            onClick={() => onCardClick(index)}
          />
        );
      })}
    </Stack>
  );
};

function App() {
  const [gameState, setGameState] = useState<GameState>({
    playerHand: Array(6).fill(''),
    tableau: Array(13).fill(''),
    suggestedMove: ''
  });

  const [selectedCard, setSelectedCard] = useState<string>('');

  const handleCardClick = (section: 'hand' | 'tableau', index: number) => {
    if (selectedCard) {
      const formattedCard = formatCardInput(selectedCard);
      setGameState(prev => ({
        ...prev,
        [section === 'hand' ? 'playerHand' : 'tableau']: 
          prev[section === 'hand' ? 'playerHand' : 'tableau'].map((card, i) => 
            i === index ? formattedCard : card
          )
      }));
      setSelectedCard('');
    }
  };

  const handleRandomHand = () => {
    setGameState(prev => ({
      ...prev,
      playerHand: generateRandomHand()
    }));
  };

  const handleQueryMCTS = async () => {
    // TODO: Implement API call to MCTS model
    console.log('Querying MCTS model...');
    setGameState(prev => ({
      ...prev,
      suggestedMove: 'Sample suggestion: Place card in position 3'
    }));
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Card Game MCTS Advisor
        </Typography>
        
        <Grid container spacing={3}>
          {/* Tableau */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom align="center">Tableau</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <TableauRow cards={gameState.tableau} onCardClick={(index) => handleCardClick('tableau', index)} startIndex={0} count={3} />
                <TableauRow cards={gameState.tableau} onCardClick={(index) => handleCardClick('tableau', index)} startIndex={3} count={5} />
                <TableauRow cards={gameState.tableau} onCardClick={(index) => handleCardClick('tableau', index)} startIndex={8} count={5} />
              </Box>
            </Paper>
          </Grid>

          {/* Player Hand */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>Your Hand</Typography>
              <Stack direction="row" spacing={1} sx={{ justifyContent: 'center' }}>
                {Array.from({ length: 6 }).map((_, index) => (
                  <PlayingCard
                    key={index}
                    value={gameState.playerHand[index] || ''}
                    isSelected={!!gameState.playerHand[index]}
                    onClick={() => handleCardClick('hand', index)}
                  />
                ))}
              </Stack>
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
                <Button 
                  variant="outlined" 
                  color="primary" 
                  onClick={handleRandomHand}
                >
                  Generate Random Hand
                </Button>
              </Box>
            </Paper>
          </Grid>

          {/* Card Selection */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h5" gutterBottom>Select a Card</Typography>
              <TextField
                fullWidth
                label="Enter card (e.g., 'king of hearts', 'queen of spades')"
                variant="outlined"
                value={selectedCard}
                onChange={(e) => setSelectedCard(e.target.value)}
                sx={{ mb: 2 }}
              />
              <Typography variant="body2" color="text.secondary">
                Click on an empty card slot to place the selected card
              </Typography>
            </Paper>
          </Grid>

          {/* MCTS Suggestion */}
          {gameState.suggestedMove && (
            <Grid item xs={12}>
              <Paper elevation={3} sx={{ p: 2, bgcolor: '#e3f2fd' }}>
                <Typography variant="h5" gutterBottom>MCTS Suggestion</Typography>
                <Typography variant="body1">{gameState.suggestedMove}</Typography>
              </Paper>
            </Grid>
          )}

          {/* Query Button */}
          <Grid item xs={12}>
            <Button 
              variant="contained" 
              color="primary" 
              fullWidth
              onClick={handleQueryMCTS}
              size="large"
            >
              Get Best Move
            </Button>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}

export default App;
