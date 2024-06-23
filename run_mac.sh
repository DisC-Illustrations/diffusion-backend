#!/bin/bash

# Überprüfen, ob venv existiert, und erstellen, falls nicht
if [ ! -d ".venv" ]; then
    echo "Erstelle virtuelle Umgebung..."
    python3 -m venv .venv
fi

# Aktivieren der virtuellen Umgebung
echo "Aktiviere virtuelle Umgebung..."
source .venv/bin/activate

# Ausführen des Python-Skripts
echo "Führe Python-Skript aus..."
# python ./app.py
export FLASK_APP=api.py
# if port is 5000: On macOS, try disabling the 'AirPlay Receiver' service from System Preferences -> General -> AirDrop & Handoff.
python -m flask run --port 8080 --host 0.0.0.0
