import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler   # <<< NUEVO
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class FeatureProcessor:
    """Procesa las features del dataset de Mango"""
    
    
    def __init__(self):
        self.label_encoders = {}
        self.embedding_matrix_train = None
        self.embedding_train_targets = None
        self.scaler = None                     # <<< NUEVO: scaler global para numéricas
        
    def parse_embedding(self, embedding_str: str) -> np.ndarray:
        """Convierte el string de embedding a array numpy"""
        if pd.isna(embedding_str):
            return np.zeros(512)  # Asumiendo 512 dimensiones
        return np.array([float(x) for x in str(embedding_str).split(',')])
    
    def create_embedding_features(self, df: pd.DataFrame, is_train: bool = True, k: int = 20) -> pd.DataFrame:
        """ Crea features basadas en los k vecinos más cercanos en el espacio de embeddings.
        - En train: usa vecinos dentro de train (leave-one-out, excluye el propio producto).
        - En test: usa vecinos en los productos de train. """
        df = df.copy()
        
        # Parsear embeddings a matriz numpy
        embeddings = np.array([self.parse_embedding(emb) for emb in df['image_embedding']])
        
        if is_train:
            # Guardamos la matriz de embeddings de train
            self.embedding_matrix_train = embeddings
            
            # Si tenemos la demanda total, la guardamos alineada con los embeddings
            if 'demand_total' in df.columns:
                self.embedding_train_targets = df['demand_total'].values
            
            # Similitud coseno entre todos los productos de train
            similarities = cosine_similarity(embeddings)  # shape (N, N)
            n = similarities.shape[0]
            
            if n > 1:
                # Excluir la similitud consigo mismo para evitar fuga de información
                np.fill_diagonal(similarities, -1.0)
                
                # k efectivo (por si hay pocos productos)
                k_eff = min(k, n - 1)
                
                # Índices de los k vecinos más similares por fila
                knn_idx = np.argpartition(-similarities, k_eff, axis=1)[:, :k_eff]
                topk_sims = np.take_along_axis(similarities, knn_idx, axis=1)  # shape (N, k_eff)
                
                # Features de similitud
                df['embedding_mean_sim_topk'] = topk_sims.mean(axis=1)
                df['embedding_max_sim_topk'] = topk_sims.max(axis=1)
                
                # Features basadas en demanda de los vecinos (si tenemos target)
                if self.embedding_train_targets is not None:
                    neighbor_demand = self.embedding_train_targets[knn_idx]  # (N, k_eff)
                    df['neighbor_mean_demand_topk'] = neighbor_demand.mean(axis=1)
                    df['neighbor_demand_sim_weighted'] = (
                        (neighbor_demand * topk_sims).sum(axis=1) / (topk_sims.sum(axis=1) + 1e-6)
                    )
            else:
                # Caso degenerado: solo un producto
                df['embedding_mean_sim_topk'] = 0.0
                df['embedding_max_sim_topk'] = 0.0
                df['neighbor_mean_demand_topk'] = df.get('demand_total', pd.Series(0.0, index=df.index))
                df['neighbor_demand_sim_weighted'] = df['neighbor_mean_demand_topk']
        
        else:
            # TEST: usamos los embeddings de train como "base"
            if self.embedding_matrix_train is not None:
                similarities = cosine_similarity(embeddings, self.embedding_matrix_train)  # (N_test, N_train)
                n_train = self.embedding_matrix_train.shape[0]
                
                if n_train > 0:
                    k_eff = min(k, n_train)
                    
                    knn_idx = np.argpartition(-similarities, k_eff, axis=1)[:, :k_eff]
                    topk_sims = np.take_along_axis(similarities, knn_idx, axis=1)  # (N_test, k_eff)
                    
                    # Features de similitud respecto a train
                    df['embedding_mean_sim_topk'] = topk_sims.mean(axis=1)
                    df['embedding_max_sim_topk'] = topk_sims.max(axis=1)
                    
                    # Features de demanda de vecinos de train (si tenemos targets de train)
                    if self.embedding_train_targets is not None:
                        neighbor_demand = self.embedding_train_targets[knn_idx]  # (N_test, k_eff)
                        df['neighbor_mean_demand_topk'] = neighbor_demand.mean(axis=1)
                        df['neighbor_demand_sim_weighted'] = (
                            (neighbor_demand * topk_sims).sum(axis=1) / (topk_sims.sum(axis=1) + 1e-6)
                        )
                    else:
                        df['neighbor_mean_demand_topk'] = 0.0
                        df['neighbor_demand_sim_weighted'] = 0.0
                else:
                    # No hay embeddings de train por algún motivo raro
                    df['embedding_mean_sim_topk'] = 0.0
                    df['embedding_max_sim_topk'] = 0.0
                    df['neighbor_mean_demand_topk'] = 0.0
                    df['neighbor_demand_sim_weighted'] = 0.0
            else:
                # Por si alguien llama al processor de test sin haber pasado antes por train
                df['embedding_mean_sim_topk'] = 0.0
                df['embedding_max_sim_topk'] = 0.0
                df['neighbor_mean_demand_topk'] = 0.0
                df['neighbor_demand_sim_weighted'] = 0.0
        
        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features temporales"""
        df = df.copy()
        
        # Convertir fechas
        df['phase_in'] = pd.to_datetime(df['phase_in'], format='%d/%m/%Y', errors='coerce')
        df['phase_out'] = pd.to_datetime(df['phase_out'], format='%d/%m/%Y', errors='coerce')
        
        # Features de fecha
        df['phase_in_month'] = df['phase_in'].dt.month
        df['phase_in_quarter'] = df['phase_in'].dt.quarter
        df['phase_out_month'] = df['phase_out'].dt.month
        
        # Temporada (basado en mes de inicio)
        df['is_summer_season'] = df['phase_in_month'].isin([3, 4, 5, 6, 7, 8]).astype(int)
        df['is_winter_season'] = df['phase_in_month'].isin([9, 10, 11, 12, 1, 2]).astype(int)
        
        return df
    
    def create_product_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Crea features de producto"""
        df = df.copy()
        
        # Features categóricas principales (Label Encoding)
        categorical_cols = [
            'aggregated_family', 'family', 'category', 'fabric',
            'length_type', 'silhouette_type', 'waist_type', 
            'neck_lapel_type', 'sleeve_length_type', 'print_type',
            'archetype', 'moment'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                if is_train:
                    # TRAIN: aseguramos que 'unknown' siempre existe como clase
                    values = df[col].fillna('unknown').astype(str)
                    
                    unique_classes = np.unique(values.tolist() + ['unknown'])
                    
                    le = LabelEncoder()
                    le.fit(unique_classes)
                    
                    df[f'{col}_encoded'] = le.transform(values)
                    self.label_encoders[col] = le
                
                else:
                    # TEST: usamos el encoder aprendido en train
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        classes_set = set(le.classes_.tolist())
                        
                        values = df[col].fillna('unknown').astype(str)
                        
                        values_mapped = values.apply(
                            lambda x: x if x in classes_set else 'unknown'
                        )
                        
                        df[f'{col}_encoded'] = le.transform(values_mapped)
                    else:
                        df[f'{col}_encoded'] = 0
        
        # One-hot encoding para color_name (solo los más comunes)
        if 'color_name' in df.columns:
            top_colors = ['NEGRO', 'BLANCO', 'AZUL', 'ROJO', 'VERDE', 'AMARILLO', 'ROSA']
            for color in top_colors:
                df[f'color_{color.lower()}'] = (df['color_name'] == color).astype(int)
        
        # Features numéricas
        df['price_log'] = np.log1p(df['price'])
        df['num_stores_log'] = np.log1p(df['num_stores'])
        
        # Interacción: stores × sizes
        df['distribution_capacity'] = df['num_stores'] * df['num_sizes']
        
        return df
    
    def aggregate_to_product_level(self, df: pd.DataFrame, mode: str = 'train') -> pd.DataFrame:
        """
        Agrega datos a nivel de producto (un registro por ID)
        Para train: calcula weekly_demand_total
        Para validación: usa solo las features del producto
        """
        if mode == 'train':
            # Agrupar por ID y calcular demanda total
            agg_dict = {}
            
            # Solo agregar las columnas que existen
            if 'weekly_demand' in df.columns:
                agg_dict['weekly_demand'] = 'sum'  # Esta será nuestra variable objetivo
            if 'weekly_sales' in df.columns:
                agg_dict['weekly_sales'] = 'sum'
            
            # Mantener las features del producto (tomar el primer valor)
            exclude_cols = ['ID', 'weekly_demand', 'weekly_sales', 'num_week_iso', 'Production']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            for col in feature_cols:
                agg_dict[col] = 'first'
            
            df_agg = df.groupby('ID', as_index=False).agg(agg_dict)
            
            # Renombrar para claridad si existe la columna
            if 'weekly_demand' in df_agg.columns:
                df_agg = df_agg.rename(columns={'weekly_demand': 'demand_total'})
            
            return df_agg
        else:
            # Para test/validación, ya viene un registro por producto
            return df
    
    def process_features(self, df: pd.DataFrame, is_train: bool = True, 
                    mode: str = 'train') -> pd.DataFrame:
        """Pipeline completo de procesamiento de features"""
        
        print(f"Procesando features - is_train: {is_train}, mode: {mode}")
        print(f"Shape inicial: {df.shape}")
        
        # Si es train, primero agregamos a nivel producto
        if mode == 'train':
            df = self.aggregate_to_product_level(df, mode='train')
            print(f"Shape después de agregación: {df.shape}")
        
        # Features temporales
        df = self.create_temporal_features(df)
        
        # Features de producto
        df = self.create_product_features(df, is_train=is_train)
        
        # Features de embeddings (similitud de coseno / kNN)
        df = self.create_embedding_features(df, is_train=is_train)
        
        # <<< NUEVO: normalización robusta de columnas numéricas >>>
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # No escalar ID, target ni Production
        num_cols = [c for c in num_cols if c not in ['ID', 'id_season', 'demand_total', 'Production']]
        
        if is_train:
            self.scaler = RobustScaler()
            if num_cols:
                df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            if self.scaler is not None and num_cols:
                df[num_cols] = self.scaler.transform(df[num_cols])
        # <<< FIN NORMALIZACIÓN >>>
        
        # Eliminar columnas categóricas originales y otras que no necesitamos
        cols_to_drop = [
            'phase_in', 'phase_out', 'image_embedding', 
            'color_name', 'color_rgb', 'weekly_sales',
            'aggregated_family', 'family', 'category', 'fabric',
            'length_type', 'silhouette_type', 'waist_type', 
            'neck_lapel_type', 'sleeve_length_type', 'print_type',
            'archetype', 'moment', 'woven_structure', 'knit_structure',
            'heel_shape_type', 'toecap_type', 'year', 'num_week_iso'
        ]
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        print(f"Shape final: {df.shape}")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Retorna las columnas de features para el modelo"""
        exclude_cols = ['ID', 'id_season', 'demand_total', 'Production']
        # Solo retornar las columnas que existen en el DataFrame
        return [col for col in df.columns if col not in exclude_cols]