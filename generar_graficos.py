import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

os.makedirs('graficos', exist_ok=True)

print("Cargando datos...")
df_no_pca = pd.read_csv('resultados_evaluacion_no_supervisada.csv')
df_pca = pd.read_csv('resultados_evaluacion_no_supervisada_pca.csv')

# 1. Comparación de anomalías detectadas por cada modelo (PCA vs No-PCA)
print("Generando gráfico de comparación PCA vs No-PCA...")
def get_anomaly_counts(df):
    return {
        'Isolation Forest': df['es_anomalia_modelo'].sum(),
        'Autoencoder': df['es_anomalia_ae'].sum(),
        'Random Forest': df['es_anomalia_rf'].sum()
    }

counts_no_pca = get_anomaly_counts(df_no_pca)
counts_pca = get_anomaly_counts(df_pca)

df_counts = pd.DataFrame([
    {'Modelo': 'Isolation Forest', 'Tipo': 'Original', 'Anomalías': counts_no_pca['Isolation Forest']},
    {'Modelo': 'Autoencoder', 'Tipo': 'Original', 'Anomalías': counts_no_pca['Autoencoder']},
    {'Modelo': 'Random Forest', 'Tipo': 'Original', 'Anomalías': counts_no_pca['Random Forest']},
    {'Modelo': 'Isolation Forest', 'Tipo': 'Con PCA', 'Anomalías': counts_pca['Isolation Forest']},
    {'Modelo': 'Autoencoder', 'Tipo': 'Con PCA', 'Anomalías': counts_pca['Autoencoder']},
    {'Modelo': 'Random Forest', 'Tipo': 'Con PCA', 'Anomalías': counts_pca['Random Forest']}
])

plt.figure(figsize=(10, 6))
sns.barplot(data=df_counts, x='Modelo', y='Anomalías', hue='Tipo', palette=['#3498db', '#e74c3c'])
plt.title('Número de Anomalías Detectadas por Modelo\n(Original vs Reducción de Dimensionalidad PCA)', fontsize=14, pad=15)
plt.ylabel('Cantidad de Anomalías', fontsize=12)
plt.xlabel('Modelo', fontsize=12)
plt.legend(title='Dataset')
plt.tight_layout()
plt.savefig('graficos/comparacion_pca_anomalias.png')
plt.close()

# 2. Distribución de puntuaciones de anomalía (Isolation Forest vs Random Forest) en No-PCA
print("Generando gráfico de distribución de scores...")
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_no_pca, x='anomalia_score', label='Isolation Forest', fill=True, color='#2ecc71')
sns.kdeplot(data=df_no_pca, x='anomalia_score_rf', label='Random Forest', fill=True, color='#9b59b6')
# Autoencoder scores are typically different scale (MSE), so we'll plot it separately or standardize
plt.title('Distribución de Puntuaciones de Anomalía\n(Isolation Forest y Random Forest)', fontsize=14, pad=15)
plt.xlabel('Puntuación de Anomalía (Score)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('graficos/distribucion_scores.png')
plt.close()

# 3. Intersección de anomalías (consenso entre modelos) en PCA
print("Generando gráfico de consenso de modelos...")
df_pca['total_votos'] = df_pca['es_anomalia_modelo'].astype(int) + df_pca['es_anomalia_ae'].astype(int) + df_pca['es_anomalia_rf'].astype(int)
votos_counts = df_pca['total_votos'].value_counts().sort_index()
# Filtrar solo donde hay al menos 1 voto para no sesgar con las transacciones normales
votos_counts = votos_counts[votos_counts.index > 0]

plt.figure(figsize=(8, 6))
ax = sns.barplot(x=votos_counts.index, y=votos_counts.values, palette='viridis')
plt.title('Consenso de Modelos en Detección de Anomalías (Dataset con PCA)', fontsize=14, pad=15)
plt.xlabel('Número de Modelos que Detectan la Anomalía', fontsize=12)
plt.ylabel('Cantidad de Transacciones', fontsize=12)

# Añadir etiquetas en las barras
for i, v in enumerate(votos_counts.values):
    ax.text(i, v + (v*0.02), str(v), ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('graficos/consenso_modelos.png')
plt.close()

# 4. Impacto del PCA en la puntuación del Autoencoder (Error de Reconstrucción)
print("Generando gráfico de impacto PCA en Autoencoder...")
plt.figure(figsize=(10, 6))
sns.kdeplot(df_no_pca['anomalia_score_ae'], label='Original', fill=True, color='#f1c40f')
sns.kdeplot(df_pca['anomalia_score_ae'], label='Con PCA', fill=True, color='#e67e22')
plt.title('Distribución del Error de Reconstrucción del Autoencoder\n(Original vs PCA)', fontsize=14, pad=15)
plt.xlabel('Error de Reconstrucción (Score)', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend(title='Dataset')
plt.tight_layout()
plt.savefig('graficos/autoencoder_reconstruccion_pca.png')
plt.close()

print("Gráficos generados exitosamente en la carpeta 'graficos/'.")
