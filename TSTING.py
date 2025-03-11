import numpy as np
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.graph_objects as go

# 📌 API Key de Alpha Vantage
ALPHA_VANTAGE_API_KEY = "6XE23J2QP58EE8L7"

# 📌 Función para descargar datos
def obtener_datos_alpha_vantage(ticker, api_key, intentos=5):
    ts = TimeSeries(key=api_key, output_format='pandas')
    for i in range(intentos):
        try:
            print(f"Descargando datos para {ticker} (Intento {i+1}/{intentos})...")
            data, _ = ts.get_daily(symbol=ticker, outputsize='full')
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.index = pd.to_datetime(data.index)
            return data.sort_index()
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
    raise Exception("No se pudieron descargar los datos.")

# 📌 Configuración
ticker = 'NFLX'
start_date = datetime.datetime(2023, 2, 19)

# 📌 Descargar datos
data = obtener_datos_alpha_vantage(ticker, ALPHA_VANTAGE_API_KEY)
data = data.loc[data.index >= start_date]

# 📌 Indicadores Técnicos
data['RSI'] = data['Close'].diff().rolling(14).mean()
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
data['%K'] = 100 * (data['Close'] - data['Low'].rolling(14).min()) / (data['High'].rolling(14).max() - data['Low'].rolling(14).min())
data['%D'] = data['%K'].rolling(3).mean()

# 📌 🚀 Índice de Capital Institucional (ICI)
data['Volume_MA_50'] = data['Volume'].rolling(window=50).mean()
data['Institutional_Index'] = data['Volume'] / data['Volume_MA_50']

# 📌 Definir Entrada y Salida de Capital
entrada_umbral = 2.0
salida_umbral = 0.5
data['Entrada_Capital'] = (data['Institutional_Index'] > entrada_umbral).astype(int)
data['Salida_Capital'] = (data['Institutional_Index'] < salida_umbral).astype(int)

# 📌 Asegurar que no haya valores NaN después de calcular los indicadores
data.fillna(method='ffill', inplace=True)
data.dropna(inplace=True)

# 📌 Selección de Características
features = ['RSI', 'MACD', 'Signal_Line', '%K', '%D', 'Volume', 'Institutional_Index']
X = data[features].values
y = data['Entrada_Capital'].values  

# 📌 Normalizar Datos
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# 📌 TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

# 📌 Aplicar SMOTE para balancear clases
if np.sum(y_train == 1) > 1:
    print("Aplicando SMOTE para balancear las clases...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

# 📌 Entrenar LSTM
model_lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model_lstm.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, verbose=1,
               validation_data=(X_test_reshaped, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

# 📌 Generar características para XGBoost
X_train_lstm_features = model_lstm.predict(X_train_reshaped).flatten()
X_test_lstm_features = model_lstm.predict(X_test_reshaped).flatten()

# 📌 Entrenar XGBoost
model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, scale_pos_weight=10)
model_xgb.fit(X_train_lstm_features.reshape(-1, 1), y_train)

y_pred = model_xgb.predict(X_test_lstm_features.reshape(-1, 1))

# 📌 Evaluación del Modelo
print(f"Precisión XGBoost: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred)}")

# 📌 📊 Gráfico de Velas Japonesas con Señales
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index, 
    open=data['Open'], 
    high=data['High'], 
    low=data['Low'], 
    close=data['Close'], 
    name='Precio'
))
fig.add_trace(go.Scatter(
    x=data.index[data['Entrada_Capital'] == 1], 
    y=data['Close'][data['Entrada_Capital'] == 1], 
    mode='markers', 
    marker=dict(size=10, color='green', symbol='triangle-up'), 
    name='Entrada de Capital'
))
fig.add_trace(go.Scatter(
    x=data.index[data['Salida_Capital'] == 1], 
    y=data['Close'][data['Salida_Capital'] == 1], 
    mode='markers', 
    marker=dict(size=10, color='blue', symbol='triangle-down'), 
    name='Salida de Capital'
))
fig.update_layout(
    title=f'Señales Institucionales para {ticker}',
    xaxis_title='Fecha',
    yaxis_title='Precio',
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    height=600,
    width=1000
)
fig.show()

# 📌 📊 Matriz de Confusión
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión - XGBoost')
plt.show()

# 📌 📊 Cálculo CORRECTO del RSI con Media Móvil Exponencial
window_rsi = 14  # Ventana estándar

delta = data['Close'].diff(1)  # Diferencia de precio día a día

# 📌 Separar ganancias y pérdidas
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)

# 📌 Media móvil exponencial de ganancias y pérdidas
avg_gain = pd.Series(gain, index=data.index).ewm(span=window_rsi, adjust=False).mean()
avg_loss = pd.Series(loss, index=data.index).ewm(span=window_rsi, adjust=False).mean()

# 📌 Evitar división por cero en `rs`
rs = avg_gain / (avg_loss + 1e-10)  # Se suma un pequeño valor para evitar errores

# 📌 Calcular el RSI
data['RSI'] = 100 - (100 / (1 + rs))

# 📌 📊 Gráfico del RSI
plt.figure(figsize=(12, 5))
plt.plot(data.index, data['RSI'], color='purple', label="RSI (14)")
plt.axhline(70, linestyle='--', color='red', alpha=0.5, label="Sobrecompra (70)")
plt.axhline(30, linestyle='--', color='green', alpha=0.5, label="Sobreventa (30)")
plt.ylim(0, 100)  # Asegurar el rango 0-100
plt.title(f'Índice de Fuerza Relativa (RSI) de {ticker}')
plt.legend()
plt.show()



# 📌 📊 Gráfico MACD
plt.figure(figsize=(12, 5))
plt.bar(data.index, data['MACD_Histogram'], color='gray', label="MACD Histogram")
plt.plot(data.index, data['MACD'], color='blue', label="MACD Line")
plt.plot(data.index, data['Signal_Line'], color='orange', label="Línea de Señal")
plt.title(f'MACD de {ticker}')
plt.legend()
plt.show()

# 📌 📊 Cálculo del Canal de Keltner (KC)
data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()  # Línea Central
data['ATR_10'] = data['High'].rolling(10).max() - data['Low'].rolling(10).min()  # Rango Verdadero Promedio
factor_kc = 2  # Factor estándar

# 📌 Bandas de Keltner
data['KC_Upper'] = data['EMA_20'] + (factor_kc * data['ATR_10'])
data['KC_Lower'] = data['EMA_20'] - (factor_kc * data['ATR_10'])

# 📌 📊 Gráfico del Canal de Keltner
plt.figure(figsize=(12, 5))
plt.plot(data.index, data['Close'], color='black', label="Precio de Cierre")
plt.plot(data.index, data['EMA_20'], color='blue', linestyle="--", label="EMA 20 (Línea Central)")
plt.plot(data.index, data['KC_Upper'], color='green', linestyle="--", label="Banda Superior KC")
plt.plot(data.index, data['KC_Lower'], color='red', linestyle="--", label="Banda Inferior KC")

plt.fill_between(data.index, data['KC_Upper'], data['KC_Lower'], color='gray', alpha=0.2)  # Relleno del canal
plt.title(f'Canal de Keltner (KC) de {ticker}')
plt.legend()
plt.show()


# 📌 Definir Señal de Compra Tradicional (Condiciones)
buy_signal = (data['RSI'] < 30) & (data['%K'] < 20) & (data['MACD'] < -5)
data['Señal_Compra'] = 0
data.loc[buy_signal, 'Señal_Compra'] = 1  # Marcar las señales de compra

# 📌 Extraer fechas y precios de las señales de compra
buy_dates = data.index[data['Señal_Compra'] == 1]
buy_prices = data['Close'][data['Señal_Compra'] == 1]

# 📌 Agregar Señales de Compra al Gráfico de Velas Japonesas
fig.add_trace(go.Scatter(
    x=buy_dates, 
    y=buy_prices, 
    mode='markers', 
    marker=dict(size=10, color='yellow', symbol='triangle-up'), 
    name='Señal de Compra'
))
