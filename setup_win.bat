:: path: setup_win.bat
@echo off
setlocal

:: (opcional) limpia venv anterior si está corrupta/abierta
if exist .venv (
  echo Eliminando .venv anterior ...
  rmdir /S /Q .venv
)

echo === creando venv ===
py -3.13 -m venv .venv 2>nul || python -m venv .venv
call .venv\Scripts\activate

echo === pip/deps ===
python -m pip install --upgrade pip
python -m pip install tensorflow==2.20.* numpy pandas scikit-learn xgboost matplotlib yfinance alpha_vantage
python -m pip install gymnasium stable-baselines3 pytest

echo === desactivar curl_cffi (evita curl:77) ===
python -m pip uninstall -y curl_cffi

echo === fija ALPHA_VANTAGE_KEY (sesion + persistente) ===
set ALPHA_VANTAGE_KEY=6JG5337BHDMKBHEI
setx ALPHA_VANTAGE_KEY 6JG5337BHDMKBHEI >nul

echo Listo.
echo Activa venv:   .\.venv\Scripts\activate
echo Entrena (Yahoo): python project\hybrid_trader.py --run ml --ticker AAPL --start 2016-01-01 --data-source yahoo --plot-bvb
echo Entrena (AV)   : python project\hybrid_trader.py --run ml --ticker AAPL --start 2016-01-01 --data-source av --plot-bvb
echo Reporte        : python project\hybrid_trader.py --run report
endlocal
