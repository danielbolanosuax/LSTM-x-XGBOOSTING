:: path: setup_win.bat
@echo off
setlocal
REM Por qué: usar siempre el python del sistema para crear venv y aislar dependencias
if "%~1"=="" (
  set PY=python
) else (
  set PY=%~1
)

echo === creando venv ===
%PY% -m venv .venv || (echo ERROR creando venv & exit /b 1)

call .venv\Scripts\activate
echo === actualizando pip ===
python -m pip install --upgrade pip

echo === instalando paquetes base ===
python -m pip install tensorflow==2.20.* numpy pandas scikit-learn xgboost yfinance matplotlib

echo === (opcional) RL ===
REM Nota: en Python 3.13 puede fallar gym/sb3. Si falla, usa py -3.11 para el venv.
python -m pip install gym==0.26.2 stable-baselines3 || echo "saltando RL (puede requerir Python 3.11)"

echo.
echo Listo. Activa el entorno con:
echo     .\.venv\Scripts\activate
echo Luego ejecuta:
echo     python project\hybrid_trader.py --run hybrid --ticker AAPL --start 2016-01-01
endlocal
