@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem === Config por defecto (cámbialos si quieres) ===
set "TICKER=AAPL"
set "START=2016-01-01"
set "SOURCE=auto"

echo.
echo ============================================
echo   LSTM-x-XGBOOSTING  -  RUN ALL
echo   Ticker     : %TICKER%
echo   Start date : %START%
echo   Data source: %SOURCE%
echo ============================================
echo.

rem === Activa venv si no esta activa ===
if not defined VIRTUAL_ENV (
  if exist ".\.venv\Scripts\activate" call ".\.venv\Scripts\activate"
)

rem === AV key visible/no visible ===
if defined ALPHA_VANTAGE_KEY (
  echo [run_all] AV_KEY detectada
) else (
  echo [run_all] AV_KEY NO detectada
)

rem === 1) Entrenamiento ===
python project\hybrid_trader.py --run ml --ticker %TICKER% --start %START% --data-source %SOURCE% --plot-bvb
if errorlevel 1 (
  echo [ERROR] Fallo en entrenamiento.
  exit /b 1
)

rem === 2) Reporte ===
python project\make_report.py
if errorlevel 1 (
  echo [ERROR] Fallo generando el reporte.
  exit /b 1
)

echo.
echo ==== FIN. (Ticker=%TICKER%  Fuente=%SOURCE%) ====
endlocal
