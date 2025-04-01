import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer el archivo CSV usando la ruta relativa
df = pd.read_csv('data/share-of-dietary-energy-supply-from-carbohydrates-vs-gdp-per-capita.csv')

# Definir nombres de columnas para facilitar su uso
gdp_column = 'GDP per capita, PPP (constant 2017 international $)'
carb_column = 'Share of calories from carbohydrates (FAO (2017))'
pop_column = 'Population (historical estimates)'
year_column = 'Year'
continent_column = 'Continent'

# 1. Gráfico de dispersión entre PIB y Share of calories from carbohydrates (filtrando solo filas con datos en ambas columnas)
df_clean = df.dropna(subset=[gdp_column, carb_column])
plt.figure(figsize=(8,6))
plt.scatter(df_clean[gdp_column], df_clean[carb_column], alpha=0.7)
plt.xlabel(gdp_column)
plt.ylabel(carb_column)
plt.title('Relación entre PIB y Share of calories from carbohydrates')
plt.grid(True)
plt.show()

# 2. Histograma del PIB per capita
plt.figure(figsize=(8,6))
plt.hist(df_clean[gdp_column], bins=30, alpha=0.7)
plt.xlabel(gdp_column)
plt.ylabel('Frecuencia')
plt.title('Distribución del PIB per capita')
plt.grid(True)
plt.show()

# 3. Histograma del Share of calories from carbohydrates
plt.figure(figsize=(8,6))
plt.hist(df_clean[carb_column], bins=30, alpha=0.7)
plt.xlabel(carb_column)
plt.ylabel('Frecuencia')
plt.title('Distribución del Share of calories from carbohydrates')
plt.grid(True)
plt.show()

# 4. Gráfico de dispersión entre PIB per capita y Población
# Filtramos filas con datos en PIB y Población
df_pop = df.dropna(subset=[gdp_column, pop_column])
plt.figure(figsize=(8,6))
plt.scatter(df_pop[gdp_column], df_pop[pop_column], alpha=0.7)
plt.xlabel(gdp_column)
plt.ylabel(pop_column)
plt.title('Relación entre PIB per capita y Población')
plt.grid(True)
plt.yscale('log')  # Escala logarítmica para la población
plt.show()

# 5. Matriz de correlación entre variables numéricas (Year, PIB, Share of calories y Población)
df_numeric = df[[year_column, gdp_column, carb_column, pop_column]].dropna()
corr = df_numeric.corr()

plt.figure(figsize=(6,5))
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
# Agregar los valores de correlación en cada celda
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')
plt.title('Matriz de correlación entre variables numéricas')
plt.tight_layout()
plt.show()

# 6. Gráfico de dispersión coloreado por Continente
# Filtramos filas con datos en PIB, Share of calories y Continente
df_cont = df.dropna(subset=[gdp_column, carb_column, continent_column])
continents = df_cont[continent_column].unique()

plt.figure(figsize=(8,6))
for cont in continents:
    subset = df_cont[df_cont[continent_column] == cont]
    plt.scatter(subset[gdp_column], subset[carb_column], alpha=0.7, label=cont)
plt.xlabel(gdp_column)
plt.ylabel(carb_column)
plt.title('Relación entre PIB y Share of calories por Continente')
plt.legend(title='Continente', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Series temporales para una entidad seleccionada (por ejemplo, "Afghanistan")
# Se filtran datos para la entidad y se eliminan filas sin datos en Año, PIB y Share of calories
entity_selected = 'Afghanistan'
df_entity = df[df['Entity'] == entity_selected].dropna(subset=[year_column, gdp_column, carb_column])

plt.figure(figsize=(8,6))
plt.plot(df_entity[year_column], df_entity[gdp_column], marker='o', label='PIB per capita')
plt.xlabel('Año')
plt.ylabel(gdp_column)
plt.title(f'Evolución del PIB per capita en {entity_selected}')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(df_entity[year_column], df_entity[carb_column], marker='o', color='orange', label='Share of calories from carbohydrates')
plt.xlabel('Año')
plt.ylabel(carb_column)
plt.title(f'Evolución del Share of calories from carbohydrates en {entity_selected}')
plt.grid(True)
plt.legend()
plt.show()

import pandas as pd

# Función auxiliar para obtener el primer valor no nulo
def get_first_nonnull(x):
    for v in x:
        if pd.notnull(v):
            return v
    return None

# Definir nombres de columnas
gdp_column = 'GDP per capita, PPP (constant 2017 international $)'
carb_column = 'Share of calories from carbohydrates (FAO (2017))'
pop_column = 'Population (historical estimates)'
year_column = 'Year'
entity_column = 'Entity'
continent_column = 'Continent'

# Agrupar datos por país: usamos el promedio para indicadores numéricos y el primer valor no nulo para el continente
df_countries = df.groupby(entity_column).agg({
    year_column: 'max',  # Tomamos el año más reciente (puede ajustarse)
    gdp_column: 'mean',
    carb_column: 'mean',
    pop_column: 'mean',
    continent_column: lambda x: get_first_nonnull(x)
}).reset_index()

# Eliminamos países sin datos en PIB o Carbohidratos
df_countries = df_countries.dropna(subset=[gdp_column, carb_column])
print(df_countries.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
continents = df_countries[continent_column].unique()
for cont in continents:
    subset = df_countries[df_countries[continent_column] == cont]
    plt.scatter(subset[gdp_column], subset[carb_column], alpha=0.7, label=cont)
plt.xlabel(gdp_column)
plt.ylabel(carb_column)
plt.title('Relación entre países: PIB vs. Carbohidratos (Promedio)')
plt.legend(title='Continente', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
for cont in continents:
    subset = df_countries[df_countries[continent_column] == cont]
    # Se usa la población para determinar el tamaño (dividido para ajustar la visualización)
    plt.scatter(subset[gdp_column], subset[carb_column],
                s=subset[pop_column] / 1e6, alpha=0.7, label=cont)
plt.xlabel(gdp_column)
plt.ylabel(carb_column)
plt.title('Relación entre países: PIB vs. Carbohidratos\nTamaño de burbuja = Población (millones)')
plt.legend(title='Continente', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

from pandas.plotting import scatter_matrix

# Seleccionar las columnas numéricas a comparar
scatter_columns = [gdp_column, carb_column, pop_column]
scatter_df = df_countries[scatter_columns].dropna()

scatter_matrix(scatter_df, figsize=(10, 8), diagonal='kde')
plt.suptitle('Matriz de dispersión entre indicadores agregados por país')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Filtrar filas sin datos en 'Share of calories from carbohydrates (FAO (2017))' y 'Continent'
df_carb = df[['Entity', 'Year', 'Share of calories from carbohydrates (FAO (2017))', 'Continent']].dropna(
    subset=['Share of calories from carbohydrates (FAO (2017))', 'Continent']
)

# Obtener la lista de continentes presentes en los datos
continentes = df_carb['Continent'].unique()

for cont in continentes:
    # Filtrar datos por continente
    df_cont = df_carb[df_carb['Continent'] == cont]
    
    # Crear tabla pivote: índices = 'Year', columnas = 'Entity', valores = 'Share of calories from carbohydrates'
    pivot_df = df_cont.pivot(index='Year', columns='Entity', 
                             values='Share of calories from carbohydrates (FAO (2017))')
    
    # Calcular la matriz de correlación entre los países (cada columna es la serie temporal de un país)
    corr_matrix = pivot_df.corr()
    
    # Visualizar la matriz de correlación
    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(ticks=range(len(corr_matrix.columns)), labels=corr_matrix.columns, rotation=90, fontsize=6)
    plt.yticks(ticks=range(len(corr_matrix.index)), labels=corr_matrix.index, fontsize=6)
    plt.title(f'Matriz de correlación entre países en {cont}\n(basada en el consumo de carbohidratos)')
    plt.tight_layout()
    plt.show()
