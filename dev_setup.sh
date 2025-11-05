#!/bin/bash
set -e

# Detect OS
OS="$(uname -s)"

# Function to install Python on Linux
install_python_linux() {
    if ! command -v python3.11 &> /dev/null; then
        echo "python3.11 not found. Installing python3.11, python3.11-venv, and python3.11-dev..."
        sudo apt-get update
        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    fi
}

# Function to check Python on Windows (assumes Git Bash or WSL)
check_python_windows() {
    if ! command -v python3.11.exe &> /dev/null && ! command -v python3.11 &> /dev/null; then
        echo "Please install Python 3.11 manually from https://www.python.org/downloads/windows/"
        exit 1
    fi
}

# Universal Python executable detection
PYTHON_EXE=""
if command -v python3.11 &> /dev/null; then
    PYTHON_EXE="python3.11"
elif command -v python3.11.exe &> /dev/null; then
    PYTHON_EXE="python3.11.exe"
elif command -v python &> /dev/null; then
    # Fallback for Windows
    PYTHON_EXE="python"
else
    echo "Python 3.11 not found. Please install Python 3.11."
    exit 1
fi

# OS-specific setup
case "$OS" in
    Linux*)
        install_python_linux
        ;;
    MINGW*|MSYS*|CYGWIN*)
        check_python_windows
        ;;
esac

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    "$PYTHON_EXE" -m venv .venv
fi
# Activate virtual environment (works for bash on Linux/macOS, Git Bash/WSL on Windows)
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Virtual environment created with Python 3.11 and packages installed."