#!/usr/bin/env bash
VENV_PATH="C:\Users\giuseppe\Documents\UniPa\Tesi\qiskit-v1.0-venv\Scripts" # percorso al venv

# attiva il venv
source "$VENV_PATH/bin/activate"

# esegui il tuo script Python (passa eventuali argomenti)
python main.py "$@"