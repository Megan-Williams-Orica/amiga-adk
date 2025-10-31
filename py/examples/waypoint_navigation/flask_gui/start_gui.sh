#!/bin/bash
# Launcher script for Flask GUI

echo "=================================="
echo "Amiga Waypoint Navigation - Web GUI"
echo "=================================="
echo ""

# Activate virtual environment
if [ -d "../../../../venv" ]; then
    echo "Activating virtual environment..."
    source ../../../../venv/bin/activate
fi

# Check if requirements are installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Flask GUI requirements..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting Flask server..."
echo "Open browser at: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run Flask app
python app.py
