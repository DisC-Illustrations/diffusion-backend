#!/bin/bash

# Überprüfen, ob venv existiert, und erstellen, falls nicht
if [ ! -d "venv" ]; then
    echo "Erstelle virtuelle Umgebung..."
    python3 -m venv venv
fi

# Aktivieren der virtuellen Umgebung
echo "Aktiviere virtuelle Umgebung..."
source venv/bin/activate

# Ausführen des Python-Skripts
echo "Führe Python-Skript aus..."
python ./app.py
