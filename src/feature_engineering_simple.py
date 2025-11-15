"""
Feature Engineering SIMPLE para predicción de demanda Mango
Sistema simplificado sin data leakage
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def extract_rgb(df):
    """
    Extrae componentes RGB del campo color_rgb
    """
    df = df.copy()
    
    if 'color_rgb' in df.columns:
        rgb = df['color_rgb'].str.extract(r'(\d+),\s*(\d+),\s*(\d+)').astype(float)
        df[['R', 'G', 'B']] = rgb
        df = df.drop(columns=['color_rgb'])
    
    return df


def one_hot_encode(df):
    """
    One-hot encoding de variables categóricas
    """
    df = df.copy()
    
    categorical_cols = [
        "aggregated_family", "family", "category", "fabric", "color_name",
        "length_type", "silhouette_type", "waist_type", "neck_lapel_type",
        "sleeve_length_type", "heel_shape_type", "toecap_type", 
        "woven_structure", "knit_structure", "print_type", 
        "archetype", "moment", "ocassion"
    ]
    
    # Solo aplicar a columnas que existen
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False, dtype=int)
    
    return df


def add_temporal_features(df):
    """
    Procesa features temporales del dataset de Mango
    Input: DataFrame con phase_in, phase_out, life_cycle_length, id_season
    Output: DataFrame con nuevas columnas temporales
    """
    df = df.copy()
    
    # Convertir a datetime (detectar formato automáticamente)
    df['phase_in'] = pd.to_datetime(df['phase_in'], errors='coerce')
    df['phase_out'] = pd.to_datetime(df['phase_out'], errors='coerce')
    
    # Features básicas de fecha
    df['month'] = df['phase_in'].dt.month
    df['quarter'] = df['phase_in'].dt.quarter
    df['week_of_year'] = df['phase_in'].dt.isocalendar().week
    df['day_of_year'] = df['phase_in'].dt.dayofyear
    
    # Estacionalidad (Spring-Summer vs Fall-Winter)
    df['is_spring_summer'] = df['month'].isin([3, 4, 5, 6, 7, 8]).astype(int)
    
    # Tendencia temporal (días desde el inicio del dataset)
    min_date = df['phase_in'].min()
    df['days_since_start'] = (df['phase_in'] - min_date).dt.days
    
    # Features de ciclo de vida
    df['weeks_to_sell'] = df['life_cycle_length']  # Ya existe, pero por claridad
    df['is_short_lifecycle'] = (df['life_cycle_length'] < 8).astype(int)
    df['is_long_lifecycle'] = (df['life_cycle_length'] > 12).astype(int)
    
    # Timing de lanzamiento (inicio, medio, fin de temporada)
    df['launch_early'] = (df['month'].isin([1, 2, 9, 10])).astype(int)  # Inicio temporada
    df['launch_mid'] = (df['month'].isin([3, 4, 11, 12])).astype(int)    # Medio temporada
    df['launch_late'] = (df['month'].isin([5, 6, 7, 8])).astype(int)     # Fin temporada
    
    # Season ID como categórica (convertir a número)
    df['season_num'] = df['id_season'].astype(int)
    
    # Interacción: lifecycle * estación (productos de verano suelen tener ciclos más cortos)
    df['lifecycle_x_season'] = df['life_cycle_length'] * df['is_spring_summer']
    
    return df


def process_image_embeddings(df, method='statistical', scaler=None, pca_model=None):
    """
    Procesa los embeddings de imagen
    
    Métodos disponibles:
    - 'keep_all': Mantiene todos los valores del embedding (512 dimensiones)
    - 'pca': Reduce a N componentes principales
    - 'mean': Promedio del vector
    - 'statistical': Estadísticas del vector (mean, std, min, max) - RECOMENDADO
    
    Returns: df, scaler, pca_model (para reutilizar en test)
    """
    df = df.copy()
    
    if 'image_embedding' not in df.columns:
        return df, scaler, pca_model
    
    # Parsear el string de embeddings a lista de floats
    if isinstance(df['image_embedding'].iloc[0], str):
        df['image_embedding_list'] = df['image_embedding'].apply(
            lambda x: [float(i) for i in x.split(',')] if pd.notna(x) and x != '' else [0]*512
        )
    else:
        df['image_embedding_list'] = df['image_embedding']
    
    # Convertir a array numpy
    embedding_matrix = np.array(df['image_embedding_list'].tolist())
    
    if method == 'keep_all':
        # Crear columnas individuales para cada dimensión
        for i in range(embedding_matrix.shape[1]):
            df[f'emb_{i}'] = embedding_matrix[:, i]
        
        # Normalizar embeddings (StandardScaler)
        if scaler is None:
            scaler = StandardScaler()
            embedding_cols = [f'emb_{i}' for i in range(embedding_matrix.shape[1])]
            df[embedding_cols] = scaler.fit_transform(df[embedding_cols])
        else:
            embedding_cols = [f'emb_{i}' for i in range(embedding_matrix.shape[1])]
            df[embedding_cols] = scaler.transform(df[embedding_cols])
    
    elif method == 'pca':
        # Reducir dimensionalidad con PCA
        if pca_model is None:
            pca_model = PCA(n_components=50)  # Reducir a 50 dimensiones
            embeddings_reduced = pca_model.fit_transform(embedding_matrix)
        else:
            embeddings_reduced = pca_model.transform(embedding_matrix)
        
        for i in range(embeddings_reduced.shape[1]):
            df[f'emb_pca_{i}'] = embeddings_reduced[:, i]
    
    elif method == 'mean':
        # Promedio del vector (1 sola dimensión)
        df['emb_mean'] = embedding_matrix.mean(axis=1)
    
    elif method == 'statistical':
        # Estadísticas del vector (4 dimensiones)
        df['emb_mean'] = embedding_matrix.mean(axis=1)
        df['emb_std'] = embedding_matrix.std(axis=1)
        df['emb_min'] = embedding_matrix.min(axis=1)
        df['emb_max'] = embedding_matrix.max(axis=1)
    
    # Eliminar columnas temporales
    df = df.drop(['image_embedding', 'image_embedding_list'], axis=1, errors='ignore')
    
    return df, scaler, pca_model


def add_interaction_features(df):
    """
    Crea features de interacción simples y útiles
    """
    df = df.copy()
    
    # Capacidad total (tiendas * tamaños)
    if 'num_stores' in df.columns and 'num_sizes' in df.columns:
        df['total_capacity'] = df['num_stores'] * df['num_sizes']
    
    # Potencial de ingresos
    if 'num_stores' in df.columns and 'price' in df.columns:
        df['revenue_potential'] = df['num_stores'] * df['price']
    
    # Precio promedio por tamaño
    if 'price' in df.columns and 'num_sizes' in df.columns:
        df['price_per_size'] = df['price'] / (df['num_sizes'] + 1)
    
    # Exposición total (semanas * tiendas)
    if 'life_cycle_length' in df.columns and 'num_stores' in df.columns:
        df['total_exposure'] = df['life_cycle_length'] * df['num_stores']
    
    # Precio * plus size (productos plus size suelen tener precios diferentes)
    if 'price' in df.columns and 'has_plus_size' in df.columns:
        df['price_x_plus'] = df['price'] * df['has_plus_size']
    
    return df


def prepare_features_simple(train_df, test_df, embedding_method='statistical'):
    """
    Pipeline completo de feature engineering SIMPLE
    
    Args:
        train_df: DataFrame de entrenamiento
        test_df: DataFrame de test
        embedding_method: 'statistical', 'pca', 'keep_all', 'mean'
    
    Returns:
        train_processed, test_processed
    """
    print("\n🔧 Feature Engineering SIMPLE...")
    
    # 1. Extraer RGB
    print("   - Extrayendo RGB...")
    train_df = extract_rgb(train_df)
    test_df = extract_rgb(test_df)
    
    # 2. Features temporales
    print("   - Creando features temporales...")
    train_df = add_temporal_features(train_df)
    test_df = add_temporal_features(test_df)
    
    # 3. Features de interacción
    print("   - Creando features de interacción...")
    train_df = add_interaction_features(train_df)
    test_df = add_interaction_features(test_df)
    
    # 4. Procesar embeddings
    print(f"   - Procesando embeddings (method={embedding_method})...")
    train_df, scaler, pca_model = process_image_embeddings(train_df, method=embedding_method)
    test_df, _, _ = process_image_embeddings(test_df, method=embedding_method, scaler=scaler, pca_model=pca_model)
    
    # 5. One-hot encoding (ÚLTIMO paso)
    print("   - Aplicando one-hot encoding...")
    train_df = one_hot_encode(train_df)
    test_df = one_hot_encode(test_df)
    
    # 6. Asegurar que train y test tengan las mismas columnas
    print("   - Alineando columnas train/test...")
    
    # Obtener columnas comunes
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    # Columnas solo en train
    only_train = train_cols - test_cols
    if only_train:
        print(f"      ⚠️  Columnas solo en train: {len(only_train)}")
        for col in only_train:
            if col not in ['Production', 'weekly_sales', 'weekly_demand']:
                test_df[col] = 0
    
    # Columnas solo en test
    only_test = test_cols - train_cols
    if only_test:
        print(f"      ⚠️  Columnas solo en test: {len(only_test)}")
        for col in only_test:
            train_df[col] = 0
    
    # Reordenar columnas de test para que coincidan con train
    feature_cols = [col for col in train_df.columns 
                    if col not in ['ID', 'Production', 'weekly_sales', 'weekly_demand', 'num_week_iso']]
    
    print(f"\n   ✅ Features totales: {len(feature_cols)}")
    print(f"   ✅ Train shape: {train_df.shape}")
    print(f"   ✅ Test shape: {test_df.shape}")
    
    return train_df, test_df, feature_cols


# Target variable
TARGET = 'Production'

