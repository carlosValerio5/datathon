"""
XGBoost con procesamiento por lotes para optimización de cobranza domiciliada
Este script implementa un flujo completo de trabajo con XGBoost utilizando procesamiento
por lotes para manejar grandes volúmenes de datos sin problemas de memoria.
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os
import shap
import time
import gc  # Garbage collector para liberar memoria

# Configuración para controlar el uso de memoria
BATCH_SIZE = 5000  # Tamaño de cada lote
MAX_SAMPLES = 50000  # Número máximo de muestras a procesar
SAMPLE_FOR_SHAP = 1000  # Muestras para análisis SHAP

# Crear directorio para resultados si no existe
os.makedirs('resultados', exist_ok=True)

# Paso 1: Conexión a DuckDB y extracción de datos por lotes
def procesar_en_lotes(query, conn, batch_size=BATCH_SIZE, max_samples=MAX_SAMPLES):
    """
    Procesa una consulta SQL en lotes para evitar problemas de memoria.
    
    Args:
        query: Consulta SQL a ejecutar
        conn: Conexión a la base de datos
        batch_size: Tamaño de cada lote
        max_samples: Número máximo de muestras a procesar
        
    Returns:
        DataFrame con los resultados combinados
    """
    print(f"Iniciando procesamiento por lotes: {query[:100]}...")
    offset = 0
    all_data = []
    total_rows = 0
    
    while True:
        batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
        print(f"Procesando lote con offset {offset}...")
        
        batch_data = conn.execute(batch_query).fetchdf()
        
        if len(batch_data) == 0:
            break
            
        all_data.append(batch_data)
        total_rows += len(batch_data)
        offset += batch_size
        
        print(f"Lote procesado. Filas acumuladas: {total_rows}")
        
        # Liberar memoria
        gc.collect()
        
        # Si alcanzamos el máximo de muestras, detenemos
        if total_rows >= max_samples:
            print(f"Alcanzado límite de {max_samples} muestras.")
            break
    
    if not all_data:
        print("No se encontraron datos.")
        return pd.DataFrame()
        
    print(f"Combinando {len(all_data)} lotes...")
    result = pd.concat(all_data, ignore_index=True)
    print(f"Datos combinados. Shape final: {result.shape}")
    
    # Liberar memoria de lotes individuales
    del all_data
    gc.collect()
    
    return result

def cargar_datos():
    """
    Carga datos desde DuckDB utilizando procesamiento por lotes.
    
    Returns:
        Tupla con DataFrames para minimización y maximización
    """
    print("Conectando a DuckDB...")
    conn = duckdb.connect('../credifiel/db/credifiel.duckdb')

    conn.execute("USE analytics")

    # Consultas para cada modelo
    query_min = """
        SELECT 
            oc.*,
            ce.idCredito,
            ce.cobro_exitoso,
            ce.tasa_recuperacion
        FROM mart_optimizacion_costos oc
        JOIN int_cobros_enriquecidos ce 
            ON oc.idBanco = ce.idBanco
    """
    
    query_max = """
        SELECT 
            mc.*,
            ce.idCredito,
            ce.cobro_exitoso,
            ce.tasa_recuperacion
        FROM analytics.mart_maximizacion_cobranza mc
        JOIN int_cobros_enriquecidos ce 
            ON mc.idBanco = ce.idBanco
    """
    
    print("Cargando datos para modelo de minimización...")
    df_minimizacion = procesar_en_lotes(query_min, conn)
    
    print("Cargando datos para modelo de maximización...")
    df_maximizacion = procesar_en_lotes(query_max, conn)
    
    print("Cerrando conexión...")
    conn.close()
    
    return df_minimizacion, df_maximizacion

# Paso 2: Preparación de datos para XGBoost
def preparar_datos(df, objetivo='cobro_exitoso'):
    """
    Prepara los datos para entrenamiento con XGBoost.
    
    Args:
        df: DataFrame con los datos
        objetivo: Columna objetivo
        
    Returns:
        Tupla con datos preparados para entrenamiento
    """
    print(f"Preparando datos para objetivo: {objetivo}")
    
    # Eliminar filas con valores nulos en columnas críticas
    df = df.dropna(subset=[objetivo])
    
    # Eliminar columnas que no son útiles para el modelo
    cols_a_eliminar = ['indice_eficiencia', 'indice_potencial_recuperacion'] 
    df = df.drop(columns=[col for col in cols_a_eliminar if col in df.columns])
    
    print(f"Shape después de limpieza: {df.shape}")
    
    # Separar características y objetivo
    X = df.drop(columns=[objetivo])
    y = df[objetivo]
    
    # Identificar columnas categóricas y numéricas
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Columnas categóricas: {len(cat_cols)}, Columnas numéricas: {len(num_cols)}")
    
    # Crear preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Conjuntos de entrenamiento: {X_train.shape}, prueba: {X_test.shape}")
    
    return X, y, X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols

# Paso 3: Entrenamiento de modelos XGBoost
def entrenar_xgboost(X_train, y_train, X_test, y_test, preprocessor, es_clasificacion=True):
    """
    Entrena un modelo XGBoost con los datos proporcionados.
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_test, y_test: Datos de prueba
        preprocessor: Preprocesador de datos
        es_clasificacion: Si es True, entrena un clasificador; si es False, un regresor
        
    Returns:
        Tupla con pipeline y modelo entrenado
    """
    print("Iniciando entrenamiento de modelo XGBoost...")
    
    # Crear modelo
    if es_clasificacion:
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
    
    # Crear pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Entrenar modelo
    print("Entrenando modelo...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Entrenamiento completado en {training_time:.2f} segundos")
    
    # Evaluar modelo
    print("Evaluando modelo...")
    y_pred = pipeline.predict(X_test)
    
    if es_clasificacion:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Guardar métricas
        with open('resultados/metricas_clasificacion.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Tiempo de entrenamiento: {training_time:.2f} segundos\n")
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        # Guardar métricas
        with open('resultados/metricas_regresion.txt', 'w') as f:
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"Tiempo de entrenamiento: {training_time:.2f} segundos\n")
    
    return pipeline, model

# Paso 4: Análisis de importancia de variables
def analizar_importancia_caracteristicas(model, X, preprocessor, cat_cols, num_cols, nombre_modelo):
    """
    Analiza y visualiza la importancia de las características del modelo.
    
    Args:
        model: Modelo XGBoost entrenado
        X: DataFrame con características
        preprocessor: Preprocesador utilizado
        cat_cols, num_cols: Listas de columnas categóricas y numéricas
        nombre_modelo: Nombre para identificar el modelo en archivos
        
    Returns:
        DataFrame con importancia de características
    """
    print(f"Analizando importancia de características para {nombre_modelo}...")
    
    # Obtener nombres de características después de one-hot encoding
    feature_names = []
    
    # Añadir nombres de características numéricas
    feature_names.extend(num_cols)
    
    # Añadir nombres de características categóricas después de one-hot encoding
    for col in cat_cols:
        # Obtener categorías únicas
        unique_values = X[col].unique()
        for val in unique_values:
            feature_names.append(f"{col}_{val}")
    
    # Obtener importancia de características
    importances = model.feature_importances_
    
    # Crear DataFrame para visualización
    if len(importances) == len(feature_names):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
    else:
        # Si hay discrepancia en la longitud, usar índices genéricos
        print(f"Advertencia: Discrepancia en longitud. Features: {len(feature_names)}, Importances: {len(importances)}")
        importance_df = pd.DataFrame({
            'Feature': [f'Feature_{i}' for i in range(len(importances))],
            'Importance': importances
        })
    
    # Ordenar por importancia
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Guardar resultados
    importance_df.to_csv(f'resultados/importancia_{nombre_modelo}.csv', index=False)
    
    # Visualizar top 20 características
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title(f'Top 20 características más importantes - {nombre_modelo}')
    plt.tight_layout()
    plt.savefig(f'resultados/importancia_{nombre_modelo}.png', dpi=300)
    plt.close()
    
    print(f"Análisis de importancia guardado en resultados/importancia_{nombre_modelo}.csv")
    
    return importance_df

# Paso 5: Análisis SHAP para interpretabilidad avanzada
def analizar_shap(model, X_train, preprocessor, nombre_modelo, num_samples=SAMPLE_FOR_SHAP):
    """
    Realiza análisis SHAP para interpretabilidad del modelo.
    
    Args:
        model: Modelo XGBoost entrenado
        X_train: Datos de entrenamiento
        preprocessor: Preprocesador utilizado
        nombre_modelo: Nombre para identificar el modelo en archivos
        num_samples: Número de muestras para análisis SHAP
        
    Returns:
        Tupla con explicador SHAP y valores SHAP
    """
    print(f"Realizando análisis SHAP para {nombre_modelo} con {num_samples} muestras...")
    
    # Tomar muestra para análisis SHAP
    if len(X_train) > num_samples:
        X_sample = X_train.sample(num_samples, random_state=42)
    else:
        X_sample = X_train
    
    # Preprocesar datos
    print("Preprocesando datos para SHAP...")
    X_sample_processed = preprocessor.transform(X_sample)
    
    # Crear explicador SHAP
    print("Creando explicador SHAP...")
    explainer = shap.TreeExplainer(model)
    
    # Calcular valores SHAP
    print("Calculando valores SHAP...")
    shap_values = explainer.shap_values(X_sample_processed)
    
    # Visualizar resumen (gráfico de barras)
    print("Generando visualizaciones SHAP...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample_processed, plot_type="bar", show=False)
    plt.title(f'SHAP - Importancia de características - {nombre_modelo}')
    plt.tight_layout()
    plt.savefig(f'resultados/shap_bar_{nombre_modelo}.png', dpi=300)
    plt.close()
    
    # Visualizar impacto detallado
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample_processed, show=False)
    plt.title(f'SHAP - Impacto de características - {nombre_modelo}')
    plt.tight_layout()
    plt.savefig(f'resultados/shap_summary_{nombre_modelo}.png', dpi=300)
    plt.close()
    
    print(f"Análisis SHAP completado y guardado en resultados/shap_*_{nombre_modelo}.png")
    
    return explainer, shap_values

# Paso 6: Creación de ecuaciones de optimización basadas en los resultados
def crear_ecuacion_optimizacion(importance_df, nombre_modelo, top_n=10):
    """
    Crea una ecuación de optimización basada en la importancia de características.
    
    Args:
        importance_df: DataFrame con importancia de características
        nombre_modelo: Nombre para identificar el modelo en archivos
        top_n: Número de características a incluir en la ecuación
        
    Returns:
        String con la ecuación
    """
    print(f"Creando ecuación de optimización para {nombre_modelo} con top {top_n} características...")
    
    # Obtener top N características
    top_features = importance_df.head(top_n)
    
    # Crear ecuación
    equation = f"Score_{nombre_modelo} = "
    
    for i, row in top_features.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        
        # Añadir término a la ecuación
        if i > 0:
            equation += " + "
        equation += f"{importance:.4f} * {feature}"
    
    # Guardar ecuación
    with open(f'resultados/ecuacion_{nombre_modelo}.txt', 'w') as f:
        f.write(equation)
    
    print(f"Ecuación guardada en resultados/ecuacion_{nombre_modelo}.txt")
    
    return equation




# Paso 7: Guardar modelos y resultados
def guardar_modelos(pipeline_min, pipeline_max, importance_min, importance_max, ecuacion_min, ecuacion_max):
    """
    Guarda modelos y resultados para uso posterior.
    
    Args:
        pipeline_min, pipeline_max: Pipelines de modelos
        importance_min, importance_max: DataFrames de importancia
        ecuacion_min, ecuacion_max: Ecuaciones de optimización
    """
    print("Guardando modelos y resultados...")
    
    # Guardar pipelines
    with open('resultados/pipeline_min.pkl', 'wb') as f:
        pickle.dump(pipeline_min, f)

    with open('resultados/pipeline_max.pkl', 'wb') as f:
        pickle.dump(pipeline_max, f)

    # Guardar DataFrames de importancia
    importance_min.to_pickle('resultados/importance_min.pkl')
    importance_max.to_pickle('resultados/importance_max.pkl')

    # Guardar ecuaciones
    with open('resultados/ecuaciones.txt', 'w') as f:
        f.write("Ecuación para minimización de costos:\n")
        f.write(ecuacion_min)
        f.write("\n\nEcuación para maximización de cobranza:\n")
        f.write(ecuacion_max)

    print("Modelos y resultados guardados correctamente en directorio 'resultados/'")


# Función principal
def main():
    """Función principal que ejecuta todo el flujo de trabajo."""
    print("Iniciando flujo de trabajo de XGBoost con procesamiento por lotes...")

    # Paso 1: Cargar datos
    df_minimizacion, df_maximizacion = cargar_datos()

    # Paso 2: Preparar datos
    print("\n--- Preparando datos para modelo de minimización de costos ---")
    X_min, y_min, X_train_min, X_test_min, y_train_min, y_test_min, preprocessor_min, cat_cols_min, num_cols_min = preparar_datos(
        df_minimizacion, 'cobro_exitoso')

    print("\n--- Preparando datos para modelo de maximización de cobranza ---")
    X_max, y_max, X_train_max, X_test_max, y_train_max, y_test_max, preprocessor_max, cat_cols_max, num_cols_max = preparar_datos(
        df_maximizacion, 'cobro_exitoso')

    # Liberar memoria de DataFrames originales
    del df_minimizacion, df_maximizacion
    gc.collect()

    # Paso 3: Entrenar modelos
    print("\n--- Entrenando modelo para minimización de costos ---")
    pipeline_min, model_min = entrenar_xgboost(X_train_min, y_train_min, X_test_min, y_test_min, preprocessor_min, True)

    print("\n--- Entrenando modelo para maximización de cobranza ---")
    pipeline_max, model_max = entrenar_xgboost(X_train_max, y_train_max, X_test_max, y_test_max, preprocessor_max, True)

    # Paso 4: Analizar importancia de características
    print("\n--- Analizando importancia de características ---")
    importance_min = analizar_importancia_caracteristicas(model_min, X_min, preprocessor_min, cat_cols_min,
                                                          num_cols_min, "minimizacion")
    importance_max = analizar_importancia_caracteristicas(model_max, X_max, preprocessor_max, cat_cols_max,
                                                          num_cols_max, "maximizacion")

    # Paso 5: Análisis SHAP
    print("\n--- Realizando análisis SHAP ---")
    explainer_min, shap_values_min = analizar_shap(model_min, X_train_min, preprocessor_min, "minimizacion")
    explainer_max, shap_values_max = analizar_shap(model_max, X_train_max, preprocessor_max, "maximizacion")

    # Paso 6: Crear ecuaciones de optimización
    print("\n--- Creando ecuaciones de optimización ---")
    ecuacion_min = crear_ecuacion_optimizacion(importance_min, "minimizacion")
    ecuacion_max = crear_ecuacion_optimizacion(importance_max, "maximizacion")

    # Paso 7: Guardar modelos y resultados
    print("\n--- Guardando modelos y resultados ---")
    guardar_modelos(pipeline_min, pipeline_max, importance_min, importance_max, ecuacion_min, ecuacion_max)

    print("\n¡Flujo de trabajo completado con éxito!")
    print(f"Todos los resultados están disponibles en el directorio 'resultados/'")

    # Paso 8: Aplicar ecuación y generar escenarios
    print("\n--- Aplicando ecuación y generando escenarios ---")

    # Usar df_max (el DataFrame de maximización de cobranza)
    df_max = X_max.copy()
    df_max['score_maximizacion'] = calcular_score(df_max)

    # Escenario A: Maximizar recuperación sin restricciones
    df_max_cobranza = df_max.sort_values(by='score_maximizacion', ascending=False)

    # Escenario B: Minimizar comisiones (limitar N envíos)
    n_envios_permitidos = 100_000
    df_min_comisiones = df_max_cobranza.head(n_envios_permitidos)

    #presupuesto_maximo = 200_000
    #df_min_comisiones = seleccionar_por_presupuesto(df_max, presupuesto_maximo)

    # Calcular monto recuperado para comparar
    monto_total_a = df_max_cobranza['monto_recuperado'].sum()
    monto_total_b = df_min_comisiones['monto_recuperado'].sum()
    eficiencia = (monto_total_b / monto_total_a) * 100 if monto_total_a > 0 else 0

    print(f"\n🔵 Escenario A: Recuperación sin límite")
    print(f"Envíos: {len(df_max_cobranza):,}, Monto recuperado: ${monto_total_a:,.2f}")

    print(f"\n🟢 Escenario B: Envíos limitados a {n_envios_permitidos}")
    print(f"Envíos: {len(df_min_comisiones):,}, Monto recuperado: ${monto_total_b:,.2f}")
    print(f"Eficiencia relativa: {eficiencia:.2f}% del escenario A")

    # Guardar ambos escenarios si deseas exportarlos
    df_max_cobranza.to_csv("resultados/escenario_maximizacion.csv", index=False)
    df_min_comisiones.to_csv("resultados/escenario_minimizacion.csv", index=False)

    # Verifica cuántos registros tienes
    n_disponibles = len(df_max)
    n_envios = min(100_000, n_disponibles)

    # Asegúrate de tener los scores calculados
    df_score = df_max.copy()
    df_score['score_maximizacion'] = calcular_score(df_score)

    # Escenario con modelo (top N)
    top_model = df_score.sort_values(by='score_maximizacion', ascending=False).head(n_envios)
    recuperado_modelo = top_model['monto_recuperado'].sum()

    # Escenario aleatorio (top N aleatorio)
    random_sample = df_score.sample(n=n_envios, random_state=42)
    recuperado_aleatorio = random_sample['monto_recuperado'].sum()

    # Comparación directa
    print("\n📊 Comparación modelo vs aleatorio")
    print(f"✅ Modelo (top {n_envios:,}):     ${recuperado_modelo:,.2f}")
    print(f"⚠️ Aleatorio ({n_envios:,}):       ${recuperado_aleatorio:,.2f}")
    print(f"🎯 Diferencia:                    ${recuperado_modelo - recuperado_aleatorio:,.2f}")
    print(f"📈 Mejora relativa:              {((recuperado_modelo / recuperado_aleatorio) - 1) * 100:.2f}%\n")

    # ---------- Gráfico de ganancia acumulada (Lift Chart) ----------

    # Ordenar por score de mayor a menor
    df_lift = df_score.sort_values(by='score_maximizacion', ascending=False).reset_index(drop=True)
    df_lift['acumulado_modelo'] = df_lift['monto_recuperado'].cumsum()
    df_lift['acumulado_aleatorio'] = df_score.sample(frac=1, random_state=42)['monto_recuperado'].cumsum().reset_index(
        drop=True)

    # Crear eje X como porcentaje de envíos realizados
    df_lift['porcentaje_envios'] = ((df_lift.index + 1) / len(df_lift)) * 100

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(df_lift['porcentaje_envios'], df_lift['acumulado_modelo'], label='Modelo (ordenado por score)',
             linewidth=2)
    plt.plot(df_lift['porcentaje_envios'], df_lift['acumulado_aleatorio'], label='Aleatorio', linestyle='--',
             linewidth=2)
    plt.xlabel('% de cobros enviados')
    plt.ylabel('Monto acumulado recuperado ($)')
    plt.title('Gráfico de ganancia acumulada (Lift Chart)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('resultados/lift_chart.png', dpi=300)
    plt.show()

def seleccionar_por_presupuesto(df, presupuesto_maximo):
    """
    Selecciona registros de cobro ordenados por score hasta alcanzar un presupuesto máximo de comisión.

    Args:
        df: DataFrame con columnas ['score_maximizacion', 'costo_estimado']
        presupuesto_maximo: límite de gasto total permitido

    Returns:
        Subset del DataFrame original con los registros más rentables dentro del presupuesto
    """
    df_ordenado = df.sort_values(by='score_maximizacion', ascending=False).copy()
    df_ordenado['costo_acumulado'] = df_ordenado['costo_estimado'].cumsum()
    df_filtrado = df_ordenado[df_ordenado['costo_acumulado'] <= presupuesto_maximo]
    return df_filtrado

def calcular_score(df):
    score = (
            0.4374 * df['tasa_recuperacion'] +
            0.3173 * df['tasa_recuperacion_monto'] +
            0.1269 * df['monto_recuperado'] -
            0.0441 * df['idBanco'] +
            0.0437 * df.get('nombre_emisora_BBVA CUENTA EN LINEA', 0) +
            0.0104 * df['idEmisora'] +
            0.0079 * df['idCredito'] +
            0.0055 * df.get('nombre_emisora_BANAMEX CUENTA', 0) +
            0.0043 * df.get('nombre_emisora_SANTANDER TRADICIONAL REINTENTO', 0) +
            0.0016 * df.get('nombre_banco_HSBC', 0)
            )
    return score


if __name__ == "__main__":
    main()
