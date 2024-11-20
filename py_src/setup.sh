#!/bin/bash

# Check if Python 3.12 is installed
if ! command -v python3.12 &> /dev/null; then
    echo "Python 3.12 is not installed. Please install it first."
    exit 1
fi

# Name of the virtual environment
VENV_NAME="venv"

# Create virtual environment using Python 3.12
python3.12 -m venv $VENV_NAME

# Activate the virtual environment
source $VENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    # Install dependencies from requirements.txt
    pip install -r requirements.txt
    echo "Dependencies installed successfully!"
else
    echo "requirements.txt not found in current directory"
    exit 1
fi

echo "Virtual environment setup complete! To activate it, run: source $VENV_NAME/bin/activate"
