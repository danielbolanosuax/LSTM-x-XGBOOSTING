@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
set "MPLBACKEND=Agg"

REM === Activa venv si no lo está ===
if not defined VIRTUAL_ENV (
  if exist ".\.venv\Scripts\activate" (
    call ".\.venv\Scripts\activate"
  ) else (
    echo [report_now] No se encontro .venv. Ejecuta setup_win.bat primero.
    pause
    exit /b 1
  )
)

REM === Ejecuta el generador de reporte técnico ===
echo [report_now] Generando reporte tecnico con project\make_report.py ...
python project\make_report.py
if errorlevel 1 (
  echo [report_now] ERROR al generar el reporte.
  pause
  exit /b 1
)

REM === Abre el HTML si existe ===
if exist "artifacts\report\report.html" (
  echo [report_now] Abriendo artifacts\report\report.html
  start "" "artifacts\report\report.html"
) else (
  echo [report_now] No se encontro artifacts\report\report.html
)

echo.
echo [report_now] OK.
pause
endlocal

