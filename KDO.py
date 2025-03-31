from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import ta
import os

# Configurar pandas para mostrar todas las filas y columnas sin truncar
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# --- CONFIGURACIÓN INICIAL ---
# API key y símbolo actualizados
api_key = '6XE23J2QP58EE8L7'
symbol = 'NVDA'

# Descargar datos históricos (diarios)
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
data.sort_index(inplace=True)

# Renombrar columnas para trabajar más cómodamente
data.rename(columns={
    '1. open': 'Open',
    '2. high': 'High',
    '3. low': 'Low',
    '4. close': 'Close',
    '5. volume': 'Volume'
}, inplace=True)

# --- CÁLCULO DE INDICADORES ---
# RSI (periodo de 14 días)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

# MACD y su señal (parámetros clásicos: 12, 26, 9)
data['MACD'] = ta.trend.macd(data['Close'], window_fast=12, window_slow=26)
data['MACD_signal'] = ta.trend.macd_signal(data['Close'], window_fast=12, window_slow=26, window_sign=9)
# Diferencia para detectar el cruce
data['MACD_diff'] = data['MACD'] - data['MACD_signal']

# MFI (Money Flow Index) con periodo de 14 días
data['MFI'] = ta.volume.money_flow_index(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14)

# ADX, +DI y -DI (con periodo de 14 días)
data['ADX'] = ta.trend.adx(high=data['High'], low=data['Low'], close=data['Close'], window=14)
data['plus_DI'] = ta.trend.adx_pos(high=data['High'], low=data['Low'], close=data['Close'], window=14)
data['minus_DI'] = ta.trend.adx_neg(high=data['High'], low=data['Low'], close=data['Close'], window=14)

# Follow-Through Day (FTD): definimos como día en que:
#   - El cierre es mayor que el cierre del día anterior
#   - El volumen supera la media móvil de 20 días del volumen
data['vol_ma20'] = data['Volume'].rolling(window=20).mean()
data['FTD'] = (data['Close'] > data['Close'].shift(1)) & (data['Volume'] > data['vol_ma20'])

# --- DEFINICIÓN DE SEÑALES (EVENTOS) ---
# Señal RSI: RSI < 30
cond_RSI = data['RSI'] < 30

# Señal MACD: cruce al alza del MACD (cuando MACD_diff pasa de negativo a positivo)
macd_cross = (data['MACD_diff'] > 0) & (data['MACD_diff'].shift(1) <= 0)

# Señal MFI: MFI < 20
cond_MFI = data['MFI'] < 20

# Señal ADX: ADX > 25 y +DI > -DI (indicativo de tendencia alcista)
cond_ADX = (data['ADX'] > 25) & (data['plus_DI'] > data['minus_DI'])

# Señal FTD: ya calculada en la columna 'FTD'
cond_FTD = data['FTD']

# Diccionario de condiciones con el nombre del indicador
condiciones = {
    'RSI': cond_RSI,
    'MACD': macd_cross,
    'MFI': cond_MFI,
    'ADX': cond_ADX,
    'FTD': cond_FTD
}

# --- SIMULACIÓN DE RETORNOS DESPUÉS DE LA SEÑAL ---
# Definir horizontes de análisis en días
horizontes = {'5': 5, '45': 45, '200': 200, '600': 600}

# Diccionario para almacenar los resultados por indicador
resultados_indicadores = {}

for indicador, condicion in condiciones.items():
    # Extraer los eventos según la condición del indicador
    eventos = data[condicion]
    eventos_resultados = []
    for fecha in eventos.index:
        # Extraer datos a partir del día de la señal
        datos_evento = data.loc[fecha:]
        # Verificar que existan suficientes datos para el mayor horizonte
        if len(datos_evento) > max(horizontes.values()):
            precio_inicial = datos_evento.iloc[0]['Close']
            # Se añade la columna 'tipo_indicador' y 'ticker' en cada registro
            registro = {
                'fecha_evento': fecha,
                'precio_inicial': precio_inicial,
                'tipo_indicador': indicador,
                'ticker': symbol
            }
            # Calcular el rendimiento para cada horizonte
            for etiqueta, dias in horizontes.items():
                precio_periodo = datos_evento.iloc[dias]['Close']
                rendimiento = (precio_periodo - precio_inicial) / precio_inicial
                registro[f'retorno_{etiqueta}'] = rendimiento
            eventos_resultados.append(registro)
    # Convertir la lista de registros en un DataFrame para el indicador actual
    resultados_indicadores[indicador] = pd.DataFrame(eventos_resultados)

# --- OUTPUT: RESULTADOS SEPARADOS POR INDICADOR ---
for indicador, df_resultados in resultados_indicadores.items():
    print(f"\nResultados para {indicador}:")
    if df_resultados.empty:
        print("  No se han encontrado eventos con esta condición.")
    else:
        print(df_resultados.to_string(index=False))

# --- EXPORTAR A .XLSX ---
# Se combinan todos los DataFrames en uno solo
df_total = pd.concat(resultados_indicadores.values(), ignore_index=True)

# Si el archivo ya existe, cargar los datos existentes y concatenarlos con los nuevos
excel_file = 'resultados_indicadores.xlsx'
if os.path.exists(excel_file):
    df_existente = pd.read_excel(excel_file)
    df_total = pd.concat([df_existente, df_total], ignore_index=True)

# Exportar a un archivo Excel sin sobrescribir los datos previos
df_total.to_excel(excel_file, index=False)
print("\nLos resultados se han guardado en 'resultados_indicadores.xlsx'.")

# --- RESUMEN ESTADÍSTICO ---
# Cargar el archivo Excel con los resultados (ya actualizado)
df_total = pd.read_excel(excel_file)

# Resumen estadístico agrupado por tipo de indicador
resumen = df_total.groupby('tipo_indicador').agg({
    'retorno_5': ['mean', 'median', 'std'],
    'retorno_45': ['mean', 'median', 'std'],
    'retorno_200': ['mean', 'median', 'std'],
    'retorno_600': ['mean', 'median', 'std'],
    'fecha_evento': 'count'  # número de eventos
})
print("\nResumen estadístico agrupado por tipo de indicador:")
print(resumen)
