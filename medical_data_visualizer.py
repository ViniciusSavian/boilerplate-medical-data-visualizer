import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Importando os dados
df = pd.read_csv('medical_examination.csv')

# 2 - Coluna overweight
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# 3 - Normalização dos dados sendo 0 sempre bom e 1 sempre ruim
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4 - Função para desenhar o gráfico categórico
def draw_cat_plot():
    # 5 - Crie um DataFrame no formato long com as colunas especificadas
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 - Agrupe e conte os dados, e renomeie as colunas para que o gráfico funcione corretamente
    df_cat['total'] = 1
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count()

    # 7 - Crie o gráfico categórico
    grid = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar")

    # 8 - Extraia a figura do FacetGrid e retorne-a
    fig = grid.fig

    # 9 - Salve o gráfico categórico
    fig.savefig('catplot.png')
    return fig

# 10 - Função para desenhar o mapa de calor
def draw_heat_map():
    # 11 - Limpe os dados, filtrando por pressões incorretas e percentis de altura e peso
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12 - Calcule a matriz de correlação
    corr = df_heat.corr()

    # 13 - Gere uma máscara para o triângulo superior da matriz de correlação
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Configuração da figura do matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15 - Trace o mapa de calor
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', square=True, cbar_kws={"shrink": .5}, ax=ax)

    # 16 - Salve o mapa de calor
    fig.savefig('heatmap.png')
    return fig