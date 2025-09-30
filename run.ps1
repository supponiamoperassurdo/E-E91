$Venv = "C:\Users\giuseppe\Documents\UniPa\Tesi\qiskit-v1.0-venv\Scripts\Activate.ps1" # Percorso dell'ambiente virtuale
if (-Not (Test-Path $Venv)) {
    Write-Error "L'ambiente virtuale non esiste. Crealo prima di eseguire questo script."
    exit 1
}
. $Venv
python .\main.py $args