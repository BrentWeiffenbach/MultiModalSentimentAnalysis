#!/bin/bash
set -e

# Check if python3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "python3.10 not found. Installing python3.10, python3.10-venv, and python3.10-dev..."
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi

# Create virtual environment with python3.10 if it doesn't already exist
if [ ! -d ".venv" ]; then
    python3.10 -m venv .venv
fi
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Virtual environment created with python3.10 and packages installed."