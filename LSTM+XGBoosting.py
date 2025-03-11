import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer  # 🔹 Manejo de NaN
from imblearn.combine import SMOTETomek  # 🔹 Mejor que SMOTE solo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

# 📌 API Key de Alpha Vantage
ALPHA_VANTAGE_API_KEY = "6XE23J2QP58EE8L7"

# 📌 Descargar Datos con Alpha Vantage
def obtener_datos_alpha_vantage(ticker, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data

# 📌 Configuración
ticker = 'MSFT'
start_date = datetime.datetime(2020, 1, 1)

print(f"Descargando datos para {ticker} desde Alpha Vantage...")
data = obtener_datos_alpha_vantage(ticker, ALPHA_VANTAGE_API_KEY)
data = data.loc[data.index >= start_date]
print(f"Datos descargados correctamente para {ticker}.")

# 📌 Indicadores Técnicos Mejorados
data['RSI'] = 100 - (100 / (1 + (data['Close'].diff().where(data['Close'].diff() > 0, 0).rolling(14).mean() /
                                 data['Close'].diff().where(data['Close'].diff() < 0, 0).abs().rolling(14).mean())))
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['%K'] = 100 * (data['Close'] - data['Low'].rolling(14).min()) / (data['High'].rolling(14).max() - data['Low'].rolling(14).min())
data['%D'] = data['%K'].rolling(3).mean()

# 📌 Filtrar datos válidos
columnas_necesarias = ['RSI', '%K', 'MACD', 'Close']
data.dropna(subset=columnas_necesarias, inplace=True)

# 📌 Definir Señal de Compra
buy_signal = (data['RSI'] < 30) & (data['%K'] < 20) & (data['MACD'] < -5)
data['Signal'] = 0
data.loc[buy_signal, 'Signal'] = 1

if data['Signal'].sum() < 2:
    raise ValueError(f"No hay suficientes señales de compra en {ticker} para entrenar el modelo.")

# 📌 Selección de Características
features = ['RSI', 'MACD', 'Signal_Line', '%K', '%D', 'Volume']
X = data[features].values
y = data['Signal'].values

# 📌 Normalizar Datos
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# 📌 TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

# 📌 Manejo de NaN antes de aplicar SMOTE
imputer = SimpleImputer(strategy="median")  # 🔹 Sustitución de NaN con la mediana
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 📌 Balanceo con `SMOTETomek`
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_imputed, y_train)

# 📌 Entrenar LSTM
model_lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

X_train_reshaped = np.reshape(X_train_balanced, (X_train_balanced.shape[0], 1, X_train_balanced.shape[1]))
X_test_reshaped = np.reshape(X_test_imputed, (X_test_imputed.shape[0], 1, X_test_imputed.shape[1]))

model_lstm.fit(X_train_reshaped, y_train_balanced, epochs=50, batch_size=32, verbose=1, validation_data=(X_test_reshaped, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

X_train_lstm_features = model_lstm.predict(X_train_reshaped).flatten()
X_test_lstm_features = model_lstm.predict(X_test_reshaped).flatten()

# 📌 Entrenar XGBoost con `scale_pos_weight`
model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, scale_pos_weight=len(y_train_balanced) / sum(y_train_balanced))
model_xgb.fit(X_train_lstm_features.reshape(-1, 1), y_train_balanced)

y_pred = model_xgb.predict(X_test_lstm_features.reshape(-1, 1))

# 📌 Evaluación
print(f"Precisión XGBoost: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión del Modelo Híbrido')
plt.show()



### VISUALIZACIONES DEL MODELO
import plotly.graph_objects as go

# 📌 Asegurar que `Predicción` está correctamente asignada antes de filtrar el último año
predicciones_df = pd.DataFrame({'Fecha': data.index[-len(y_pred):], 'Predicción': y_pred})
predicciones_df.set_index('Fecha', inplace=True)

# 📌 Fusionar las predicciones con el DataFrame original `data`
data = data.merge(predicciones_df, left_index=True, right_index=True, how="left")

# 📌 Filtrar datos de los últimos 365 días correctamente
one_year_ago = data.index.max() - pd.DateOffset(days=365)
data_last_year = data.loc[data.index >= one_year_ago]

# 📌 Verificar si `Predicción` está presente en `data_last_year`
if 'Predicción' not in data_last_year.columns:
    raise KeyError("La columna 'Predicción' no se encontró en los datos filtrados.")

# 📌 Obtener fechas y precios de las señales de compra
buy_dates = data_last_year.index[data_last_year['Predicción'] == 1]
buy_prices = data_last_year['Close'][data_last_year['Predicción'] == 1]

# 📌 Crear gráfico de velas con `plotly`
fig = go.Figure()

# 🔹 Agregar velas japonesas
fig.add_trace(go.Candlestick(
    x=data_last_year.index, 
    open=data_last_year['Open'], 
    high=data_last_year['High'], 
    low=data_last_year['Low'], 
    close=data_last_year['Close'], 
    name='Precio'
))

# 🔹 Agregar señales de compra en amarillo
fig.add_trace(go.Scatter(
    x=buy_dates, 
    y=buy_prices, 
    mode='markers', 
    marker=dict(size=10, color='yellow', symbol='triangle-up'), 
    name='Señal de Compra'
))

# 🔹 Configuración del diseño del gráfico
fig.update_layout(
    title=f'Señales de Compra para {ticker} - Último Año',
    xaxis_title='Fecha',
    yaxis_title='Precio',
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    height=600,
    width=1000
)

# 📌 Mostrar gráfico interactivo
fig.show()


