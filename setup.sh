#!/bin/bash
# Crypto Prediction Bot — One-time setup
# Run this once: bash setup.sh
# Then to start the bot: source venv/bin/activate && python main.py

echo "Setting up Crypto Prediction Bot..."

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo ""
echo "========================================="
echo "  Setup complete!"
echo "  To run the bot:"
echo "    source venv/bin/activate"
echo "    python main.py"
echo "========================================="
