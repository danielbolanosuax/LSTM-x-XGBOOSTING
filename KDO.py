from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import ta  # Asegúrate de tener instalada la librería: pip install ta

# Reemplaza 'YOUR_API_KEY' por tu clave de API de Alpha Vantage
api_key = '6XE23J2QP58EE8L7'
symbol = 'AAPL'

# 1. Descargar datos históricos usando Alpha Vantage
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')

# Ordenar el DataFrame por fecha (ascendente)
data.sort_index(inplace=True)

# Renombrar columnas para facilitar el uso
data.rename(columns={
    '1. open': 'Open',
    '2. high': 'High',
    '3. low': 'Low',
    '4. close': 'Close',
    '5. volume': 'Volume'
}, inplace=True)

# 2. Calcular el RSI con un período de 14 días
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

# 3. Identificar eventos: Fechas en las que el RSI es menor a 30
eventos = data[data['RSI'] < 30]

# 4. Definir los horizontes de análisis (por ejemplo, 5, 45, 200 y 600 días)
horizontes = {'5': 5, '45': 45, '200': 200, '600': 600}

# Lista para almacenar los resultados de cada evento
resultados = []

for fecha in eventos.index:
    datos_evento = data.loc[fecha:]
    # Verificar que existan suficientes datos para el mayor horizonte
    if len(datos_evento) >= max(horizontes.values()):
        precio_inicial = datos_evento.iloc[0]['Close']
        registro = {'fecha_evento': fecha, 'precio_inicial': precio_inicial}
        # Calcular el retorno en cada horizonte
        for etiqueta, dias in horizontes.items():
            precio_periodo = datos_evento.iloc[dias]['Close']
            retorno = (precio_periodo - precio_inicial) / precio_inicial
            registro[f'retorno_{etiqueta}'] = retorno
        resultados.append(registro)

# Convertir la lista de resultados en un DataFrame
df_resultados = pd.DataFrame(resultados)

# 5. Calcular estadísticas: media de retornos y % de eventos con subida/bajada para cada horizonte
estadisticas = {}
for etiqueta in horizontes.keys():
    col = f'retorno_{etiqueta}'
    media = df_resultados[col].mean()
    porcentaje_subida = (df_resultados[col] > 0).mean() * 100  # % de eventos positivos
    porcentaje_bajada = (df_resultados[col] < 0).mean() * 100   # % de eventos negativos
    estadisticas[etiqueta] = {
        'media': media,
        'porcentaje_subida': porcentaje_subida,
        'porcentaje_bajada': porcentaje_bajada,
        'n_eventos': df_resultados.shape[0]
    }

# Mostrar las estadísticas
print("Estadísticas de retornos en distintos horizontes:")
for horizonte, stats in estadisticas.items():
    print(f"\nHorizonte {horizonte} días:")
    print(f"  - Media: {stats['media']:.2%}")
    print(f"  - % Subida: {stats['porcentaje_subida']:.2f}%")
    print(f"  - % Bajada: {stats['porcentaje_bajada']:.2f}%")
    print(f"  - Número de eventos analizados: {stats['n_eventos']}")
