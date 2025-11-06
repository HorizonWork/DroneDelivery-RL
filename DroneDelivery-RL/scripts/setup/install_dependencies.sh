#!/bin/bash
# Install all dependencies

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing package in development mode..."
pip install -e .

echo "Dependencies installed successfully!"
