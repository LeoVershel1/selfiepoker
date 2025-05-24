#!/bin/bash

# Start the backend
cd src
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py &
BACKEND_PID=$!

# Start the frontend
cd ../frontend
npm start &
FRONTEND_PID=$!

# Function to handle script termination
cleanup() {
    echo "Shutting down..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

# Set up trap to catch termination signal
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait 