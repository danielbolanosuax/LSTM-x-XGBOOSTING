:: path: setup_win.bat
@echo off
setlocal
:: Por qué: 1-clic para entorno + clave AV + deps
if "%~1"=="" ( set PY=python ) else ( set PY=%~1 )

echo === creando venv ===
%PY% -m venv .venv || (echo ERROR venv & exit /b 1)
call .venv\Scripts\activate

echo === pip/deps ===
python -m pip install --upgrade pip
python -m pip install tensorflow==2.20.* numpy pandas scikit-learn xgboost matplotlib yfinance alpha_vantage
python -m pip install gymnasium stable-baselines3 pytest

echo === fijando Alpha Vantage KEY ===
:: Reemplaza si quieres otra; usaremos la que nos diste ahora
setx ALPHA_VANTAGE_KEY 6JG5337BHDMKBHEI

echo Listo. Ejemplo:
echo   python project\hybrid_trader.py --run ml --ticker AAPL --start 2016-01-01 --data-source av --plot-bvb
echo   python project\hybrid_trader.py --run report
endlocal
