REM Überprüfen, ob venv existiert, und erstellen, falls nicht
if not exist venv (
    echo Erstelle virtuelle Umgebung...
    python -m venv venv
)

REM Aktivieren der virtuellen Umgebung
echo Aktiviere virtuelle Umgebung...
call venv\Scripts\activate.bat

REM Ausführen des Python-Skripts
echo Führe Python-Skript aus...
python .\app.py
