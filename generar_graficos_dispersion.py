import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

os.makedirs('graficos', exist_ok=True)

print("Cargando datos...")
df = pd.read_csv('resultados_evaluacion_no_supervisada.csv')

# Preprocesamiento rápido para PCA en 2D (usamos variables numéricas principales)
variables_numericas = ['TotalBet', 'TotalWin', 'BalanceChange', 'ratio_ganancia']
# Rellenamos nulos por si acaso
X = df[variables_numericas].fillna(0)

# Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA a 2 componentes
pca = PCA(n_components=2)
componentes = pca.fit_transform(X_scaled)
df['PCA1'] = componentes[:, 0]
df['PCA2'] = componentes[:, 1]

# Identificar anomalías por consenso (al menos 2 modelos dicen que es anomalía)
df['votos_anomalia'] = df['es_anomalia_modelo'].astype(int) + df['es_anomalia_ae'].astype(int) + df['es_anomalia_rf'].astype(int)
df['es_anomalia_consenso'] = df['votos_anomalia'] >= 2

# 1. Gráfico de dispersión con PCA marcando anomalías
print("Generando gráfico de dispersión PCA...")
plt.figure(figsize=(10, 8))
# Trazar puntos normales
sns.scatterplot(
    data=df[~df['es_anomalia_consenso']], 
    x='PCA1', y='PCA2', 
    color='lightgray', alpha=0.5, label='Normal', s=20
)
# Trazar anomalías por consenso
sns.scatterplot(
    data=df[df['es_anomalia_consenso']], 
    x='PCA1', y='PCA2', 
    color='#e74c3c', alpha=0.8, label='Anomalía (Consenso)', s=40
)
plt.title('Proyección PCA 2D del Dataset de Transacciones\nResaltando Anomalías Detectadas', fontsize=14, pad=15)
plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} varianza)', fontsize=12)
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%} varianza)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('graficos/dispersion_pca_anomalias.png')
plt.close()

# 2. Gráfico de dispersión TotalBet vs TotalWin
print("Generando gráfico de dispersión TotalBet vs TotalWin...")
plt.figure(figsize=(10, 6))
# Para evitar solapamiento masivo, tomamos una muestra si el dataset es muy grande, o usamos alpha
sns.scatterplot(
    data=df[~df['es_anomalia_consenso']], 
    x='TotalBet', y='TotalWin', 
    color='#3498db', alpha=0.3, label='Normal', s=20
)
sns.scatterplot(
    data=df[df['es_anomalia_consenso']], 
    x='TotalBet', y='TotalWin', 
    color='#e74c3c', alpha=0.8, label='Anomalía', s=50
)
plt.title('Relación entre Apuesta Total (TotalBet) y Ganancia (TotalWin)', fontsize=14, pad=15)
plt.xlabel('Total Bet (Monto Apostado)', fontsize=12)
plt.ylabel('Total Win (Monto Ganado)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('graficos/dispersion_bet_vs_win.png')
plt.close()

# 3. Gráfico de dispersión interactivo estilo Heatmap (Densidad 2D para ver concentración)
print("Generando gráfico de densidad 2D...")
plt.figure(figsize=(10, 8))
sns.histplot(
    data=df, 
    x='PCA1', y='PCA2', 
    cmap="mako", fill=True, bins=50, pthresh=.1
)
# Superponer anomalías de Isolation Forest como puntos
sns.scatterplot(
    data=df[df['es_anomalia_modelo']], 
    x='PCA1', y='PCA2', 
    color='red', marker='+', label='Anomalía (Isolation Forest)', s=30, alpha=0.6
)
plt.title('Mapa de Densidad del Dataset (PCA) y\nAnomalías de Isolation Forest', fontsize=14, pad=15)
plt.xlabel('Componente Principal 1', fontsize=12)
plt.ylabel('Componente Principal 2', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('graficos/densidad_pca_isolation_forest.png')
plt.close()

# 4. Pairplot de algunas variables clave (Muestra aleatoria de 2000 transacciones para rendimiento)
print("Generando Pairplot (Muestra del dataset)...")
df_sample = df.sample(n=min(2000, len(df)), random_state=42)
df_sample['Etiqueta'] = df_sample['es_anomalia_consenso'].map({True: 'Anomalía', False: 'Normal'})
pairplot_cols = ['TotalBet', 'TotalWin', 'ratio_ganancia', 'anomalia_score', 'Etiqueta']
g = sns.pairplot(
    df_sample[pairplot_cols], 
    hue='Etiqueta', 
    palette={'Normal': '#3498db', 'Anomalía': '#e74c3c'},
    plot_kws={'alpha': 0.6, 's': 20},
    diag_kws={'fill': True}
)
g.fig.suptitle('Matriz de Dispersión (Pairplot) de Variables Clave', y=1.02, fontsize=14)
plt.savefig('graficos/pairplot_variables.png')
plt.close()

print("Nuevos gráficos generados en 'graficos/'.")
