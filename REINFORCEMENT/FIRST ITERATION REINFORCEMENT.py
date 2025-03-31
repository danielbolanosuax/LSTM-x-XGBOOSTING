import pandas as pd
import numpy as np
import time
import gym
from stable_baselines3 import PPO
import xgboost as xgb
from alpha_vantage.timeseries import TimeSeries

# Para el modelo LSTM usaremos Keras (TensorFlow)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
# Si deseas usar SMOTE u otras técnicas, recuerda importarlas (por ejemplo, de imblearn)

# ─────────────────────────────────────────────
# 1. DESCARGA DE DATOS CON ALPHA VANTAGE
# ─────────────────────────────────────────────

ALPHA_VANTAGE_API_KEY = "6XE23J2QP58EE8L7"

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

# ─────────────────────────────────────────────
# 2. CÁLCULO DE INDICADORES TÉCNICOS
# ─────────────────────────────────────────────

def calcular_indicadores(data):
    # Aquí se implementarían los cálculos reales de indicadores (RSI, MACD, etc.)
    # Para este ejemplo se añaden columnas dummy
    data['RSI'] = 50  # Valor fijo de ejemplo
    data['MACD'] = 0  # Valor fijo de ejemplo
    data['Stochastic_%K'] = 0  # Valor fijo de ejemplo
    data['Stochastic_%D'] = 0  # Valor fijo de ejemplo
    # OBV: ejemplo simple (no es la fórmula real)
    data['OBV'] = data['Volume'].cumsum()
    # Índice Institucional: volumen actual / media móvil de 50 días del volumen
    data['Volumen_50d'] = data['Volume'].rolling(window=50).mean()
    data['Indice_Institucional'] = data['Volume'] / data['Volumen_50d']
    data = data.dropna()  # Eliminar filas con valores nulos
    return data

# ─────────────────────────────────────────────
# 3. ETIQUETADO DE LOS DATOS
# ─────────────────────────────────────────────

def etiquetar_datos(data):
    # Se define una etiqueta 1 si el índice institucional supera un umbral (por ejemplo, 1.5)
    data['Entrada_Capital'] = (data['Indice_Institucional'] > 1.5).astype(int)
    return data

# ─────────────────────────────────────────────
# 4. PREPROCESAMIENTO DE DATOS
# ─────────────────────────────────────────────

def preprocesar_datos(data):
    scaler = MinMaxScaler()
    # Escalar todas las columnas excepto la etiqueta 'Entrada_Capital'
    features = data.drop(['Entrada_Capital'], axis=1)
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
    df_scaled['Entrada_Capital'] = data['Entrada_Capital']
    return df_scaled

# ─────────────────────────────────────────────
# 5. ENTRENAMIENTO DEL MODELO LSTM
# ─────────────────────────────────────────────

def entrenar_lstm(data_train):
    # Preparamos datos para la LSTM usando la columna 'Close' como secuencia
    sequence_length = 10  # Número de timesteps
    X, y = [], []
    prices = data_train['Close'].values
    for i in range(len(prices) - sequence_length):
        X.append(prices[i:i+sequence_length])
        y.append(prices[i+sequence_length])
    X = np.array(X)
    y = np.array(y)
    # Reshape para LSTM: [muestras, timesteps, características]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Construcción de un modelo LSTM simple
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Entrenamiento (se usan pocas épocas para el ejemplo)
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    
    # Obtener características internas de la LSTM para cada muestra.
    # En este ejemplo usamos la predicción final como "feature" extra.
    lstm_features = model.predict(X)
    # Ajustamos la longitud para que coincida con data_train: rellenamos los primeros 'sequence_length' valores con 0.
    lstm_features_full = np.zeros((len(data_train), 1))
    lstm_features_full[sequence_length:] = lstm_features
    return lstm_features_full, model

# ─────────────────────────────────────────────
# 6. ENTRENAMIENTO DEL MODELO XGBOOST
# ─────────────────────────────────────────────

def entrenar_xgboost(data_train, lstm_features):
    # Extraer características originales (sin la etiqueta) y combinar con las salidas de la LSTM.
    X_train = data_train.drop(['Entrada_Capital'], axis=1).values
    X_train = np.concatenate([X_train, lstm_features], axis=1)
    y_train = data_train['Entrada_Capital'].values
    
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model

# ─────────────────────────────────────────────
# 7. DEFINICIÓN DEL ENTORNO DE TRADING PARA RL
# ─────────────────────────────────────────────

class TradingEnv(gym.Env):
    def __init__(self, data, xgb_model, lstm_model):
        super(TradingEnv, self).__init__()
        # Espacio de acciones: 0 = Mantener, 1 = Comprar, 2 = Vender
        self.action_space = gym.spaces.Discrete(3)
        # Número de features usado en el entrenamiento de XGBoost (data sin etiqueta + 1 de LSTM)
        num_features = data.drop(columns=['Entrada_Capital']).shape[1] + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.xgb_model = xgb_model
        self.lstm_model = lstm_model
        self.done = False
        self.balance = 10000  # Balance inicial
        self.position = 0     # Cantidad de acciones en cartera

    def step(self, action):
        # Obtener observación actual: usamos los datos de mercado SIN la etiqueta
        obs_features = self.data.iloc[self.current_step].drop('Entrada_Capital').values.astype(np.float32)
        # Calcular señal del ensemble a partir del modelo LSTM y XGBoost
        ensemble_signal = self.get_ensemble_signal()
        # El estado es la concatenación de las características de mercado y la señal del ensemble
        state = np.concatenate([obs_features, [ensemble_signal]])
        
        # Simular la transacción según la acción tomada
        reward = self.simular_transaccion(action, obs_features)
        
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        return state, reward, self.done, {}

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.done = False
        obs_features = self.data.iloc[self.current_step].drop('Entrada_Capital').values.astype(np.float32)
        ensemble_signal = self.get_ensemble_signal()
        state = np.concatenate([obs_features, [ensemble_signal]])
        return state

    def get_ensemble_signal(self):
        # Usar la fila actual de datos, pero solamente las características (sin la etiqueta)
        features = self.data.iloc[self.current_step].drop('Entrada_Capital').values.astype(np.float32)
        # Para la LSTM, usamos de forma dummy el valor "Close" (asumimos que es la cuarta columna)
        # Asegúrate de que el orden de columnas sea: ['Open', 'High', 'Low', 'Close', ...]
        dummy_sequence = np.array([features[3]]).reshape(1, 1, 1)
        lstm_pred = self.lstm_model.predict(dummy_sequence)[0][0]
        # Combinar las features con la predicción de la LSTM
        combined = np.concatenate([features, [lstm_pred]]).reshape(1, -1)
        ensemble_pred = self.xgb_model.predict(combined)
        return ensemble_pred[0]

    def simular_transaccion(self, action, obs_features):
        # Suponiendo que la primera característica es el precio
        precio = obs_features[0]
        if action == 1:  # Comprar
            self.position += 1
            self.balance -= precio
        elif action == 2 and self.position > 0:  # Vender
            self.position -= 1
            self.balance += precio
        patrimonio = self.balance + self.position * precio
        reward = patrimonio - 10000  # Variación respecto al balance inicial
        return reward

# ─────────────────────────────────────────────
# 8. ENTRENAMIENTO DEL AGENTE RL CON PPO
# ─────────────────────────────────────────────

def entrenar_agente_rl(data, xgb_model, lstm_model):
    env = TradingEnv(data, xgb_model, lstm_model)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model

# ─────────────────────────────────────────────
# 9. FLUJO PRINCIPAL (MAIN)
# ─────────────────────────────────────────────

def main():
    ticker = "PLTR"
    
    # Paso 1: Descarga de datos usando Alpha Vantage
    data = obtener_datos_alpha_vantage(ticker, ALPHA_VANTAGE_API_KEY)
    
    # Paso 2: Cálculo de indicadores técnicos
    data = calcular_indicadores(data)
    
    # Paso 3: Etiquetado de datos para identificar entrada de capital institucional
    data = etiquetar_datos(data)
    
    # Paso 4: Preprocesamiento (normalización y ajustes)
    data = preprocesar_datos(data)
    
    # Dividir datos en entrenamiento y prueba (80% / 20%)
    train_data = data.iloc[:int(0.8 * len(data))]
    test_data = data.iloc[int(0.8 * len(data)):]
    
    # Paso 5: Entrenar la LSTM sobre los datos de entrenamiento
    lstm_features_train, lstm_model = entrenar_lstm(train_data)
    
    # Paso 6: Entrenar el modelo XGBoost combinando la salida de la LSTM y las características originales
    xgb_model = entrenar_xgboost(train_data, lstm_features_train)
    
    # Paso 7: Entrenar el agente RL (usando PPO) en el entorno con los datos de prueba
    rl_model = entrenar_agente_rl(test_data, xgb_model, lstm_model)
    
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()

