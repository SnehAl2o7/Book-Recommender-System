#!/bin/bash

echo "🚀 Starting Book Recommender System..."
echo ""
echo "Starting FastAPI backend server..."
echo "The backend will be available at: http://localhost:8000"
echo ""
echo "Note: First-time startup will take a few minutes to build the vector database."
echo "Once you see '🎉 Book Recommender System is ready!', open index.html in your browser."
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

~/.conda/envs/genai/bin/python backend.py
