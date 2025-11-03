@echo off
setlocal
chcp 65001 >nul

REM Activa venv si hace falta
if not defined VIRTUAL_ENV (
  if exist ".\.venv\Scripts\activate" (
    call ".\.venv\Scripts\activate"
  ) else (
    echo [patch_and_report] No se encontro .venv. Ejecuta setup_win.bat primero.
    pause
    exit /b 1
  )
)

echo [patch_and_report] Añadiendo y_true a artifacts\probabilities_ml.csv ...
python project\add_ytrue_from_cache.py
if errorlevel 1 (
  echo [patch_and_report] ERROR al crear y_true.
  pause
  exit /b 1
)

echo [patch_and_report] Generando reporte completo...
python project\make_report.py
if errorlevel 1 (
  echo [patch_and_report] ERROR generando el reporte.
  pause
  exit /b 1
)

if exist "artifacts\report\report.html" (
  start "" "artifacts\report\report.html"
)
echo [patch_and_report] OK.
pause
endlocal
