# Launch the Madrid appraisal web app locally.
# The base Python 3.12 install is missing venv launchers, so we use it directly.
$ErrorActionPreference = "Stop"
$py = "C:\Users\IVANL\AppData\Local\Programs\Python\Python312\python.exe"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path (Join-Path $here "..\best_lgbm.pkl"))) {
    Write-Host "Warning: best_lgbm.pkl not found in project root. The app will start but predictions stay disabled until it exists." -ForegroundColor Yellow
}

Write-Host "Starting server at http://localhost:8000  (Ctrl+C to stop)" -ForegroundColor Green
Push-Location $here
& $py -m uvicorn app:app --host 127.0.0.1 --port 8000
Pop-Location
