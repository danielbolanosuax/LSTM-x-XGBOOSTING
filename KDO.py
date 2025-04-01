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
symbol = 'ABT'  

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

# Precio final (último precio disponible en el histórico)
precio_final = data.iloc[-1]['Close']
fecha_final = data.index[-1]  # Fecha final utilizada para el cálculo del retorno_general

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
            # Se añade la columna 'tipo_indicador', 'ticker', 'fecha_evento' y 'fecha_final' en cada registro
            registro = {
                'fecha_evento': fecha,
                'fecha_final': fecha_final,
                'precio_inicial': precio_inicial,
                'tipo_indicador': indicador,
                'ticker': symbol
            }
            # Calcular el rendimiento para cada horizonte
            for etiqueta, dias in horizontes.items():
                precio_periodo = datos_evento.iloc[dias]['Close']
                rendimiento = (precio_periodo - precio_inicial) / precio_inicial
                registro[f'retorno_{etiqueta}'] = rendimiento
            # Calcular el rendimiento general desde el momento de la operación hasta el último precio disponible
            registro['retorno_general'] = (precio_final - precio_inicial) / precio_inicial
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

# Resumen estadístico agrupado por tipo de indicador, incluyendo retorno_general
resumen = df_total.groupby('tipo_indicador').agg({
    'retorno_5': ['mean', 'median', 'std'],
    'retorno_45': ['mean', 'median', 'std'],
    'retorno_200': ['mean', 'median', 'std'],
    'retorno_600': ['mean', 'median', 'std'],
    'retorno_general': ['mean', 'median', 'std'],
    'fecha_evento': 'count'  # número de eventos
})

# Calcular el %VAR para retorno_general como (std/mean)*100
resumen[('retorno_general', '%var')] = resumen[('retorno_general', 'std')] / resumen[('retorno_general', 'mean')] * 100

print("\nResumen estadístico agrupado por tipo de indicador:")
print(resumen)

# --- TABLA DE RENDIMIENTOS MENSUALES POR AÑO ---

# Asegurarnos de que el índice del DataFrame sea de tipo datetime
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)

# 1. Obtenemos el último precio de cierre de cada mes
monthly_close = data['Close'].resample('M').last()

# 2. Calculamos el rendimiento mensual (pct_change() da el porcentaje de variación)
monthly_returns = monthly_close.pct_change().dropna()  # El primer mes será NaN, por eso dropna()

# 3. Convertimos a DataFrame y extraemos columnas de año y mes
df_monthly = pd.DataFrame({'MonthlyReturn': monthly_returns})
df_monthly['Year'] = df_monthly.index.year
df_monthly['Month'] = df_monthly.index.month

# 4. Creamos la tabla dinámica con los años como índice y los meses como columnas
monthly_pivot = df_monthly.pivot(index='Year', columns='Month', values='MonthlyReturn')

# Renombramos las columnas numéricas del 1 al 12 por los nombres de meses en español
nombre_meses = ['ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DIC']
monthly_pivot.columns = nombre_meses[:len(monthly_pivot.columns)]

# 5. Agregamos una columna 'TOTAL' con el rendimiento acumulado anual
#    Para ello, se multiplica (1 + cada retorno mensual) y luego se resta 1
monthly_pivot['TOTAL'] = (1 + monthly_pivot).prod(axis=1) - 1

# 6. Formateamos los datos como porcentajes
monthly_pivot = monthly_pivot.fillna(0)  # Si hay meses sin datos, rellenamos con 0
formatted_pivot = monthly_pivot.applymap(lambda x: f"{x*100:.2f}%")

# 7. Imprimimos la tabla final
print("\nTabla de rendimientos mensuales por año:")
print(formatted_pivot)

# --- GUARDAR TABLA DE RENDIMIENTOS MENSUALES EN HISTORICO_TICKERS.XLSX ---

# Convertimos el pivot (monthly_pivot) en un DataFrame y reiniciamos el índice para que 'Year' sea una columna.
df_hist = monthly_pivot.reset_index()
df_hist.rename(columns={'index': 'Año'}, inplace=True)  # En caso de que el índice se llame 'index'

# Si la columna 'TOTAL' existe y no se requiere, la eliminamos
if 'TOTAL' in df_hist.columns:
    df_hist = df_hist.drop(columns=['TOTAL'])

# Renombramos la columna de año a 'Año' (por si no lo estuviera) y añadimos la columna 'Ticker'
df_hist.rename(columns={'Year': 'Año'}, inplace=True)
df_hist.insert(1, 'Ticker', symbol)

# Aseguramos el orden de las columnas: AÑO, TICKER, y luego los meses de ENE a DIC
columnas_deseadas = ['Año', 'Ticker', 'ENE', 'FEB', 'MAR', 'ABR', 'MAY', 'JUN', 'JUL', 'AGO', 'SEP', 'OCT', 'NOV', 'DIC']
df_hist = df_hist.reindex(columns=columnas_deseadas)

# Si ya existe el archivo, cargamos los datos existentes y los concatenamos con los nuevos.
historico_file = 'historico_tickers.xlsx'
if os.path.exists(historico_file):
    df_existente = pd.read_excel(historico_file)
    df_combined = pd.concat([df_existente, df_hist], ignore_index=True)
else:
    df_combined = df_hist

# Guardamos el DataFrame combinado en el archivo Excel.
df_combined.to_excel(historico_file, index=False)
print("\nLa tabla de rendimientos mensuales se ha guardado en 'historico_tickers.xlsx'.")
