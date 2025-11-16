import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class FeatureProcessor:
    """Procesa las features del dataset de Mango"""
    
    def __init__(self):
        self.label_encoders = {}
        self.embedding_matrix_train = None
        
    def parse_embedding(self, embedding_str: str) -> np.ndarray:
        """Convierte el string de embedding a array numpy"""
        if pd.isna(embedding_str):
            return np.zeros(512)  # Asumiendo 512 dimensiones
        return np.array([float(x) for x in str(embedding_str).split(',')])
    
    def create_embedding_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Crea features de similitud de coseno con los embeddings de imágenes
        """
        df = df.copy()
        
        # Parsear embeddings
        embeddings = np.array([self.parse_embedding(emb) for emb in df['image_embedding']])
        
        if is_train:
            self.embedding_matrix_train = embeddings
            # Para train, calculamos similitud promedio con todos los productos
            similarities = cosine_similarity(embeddings)
            df['embedding_avg_similarity'] = similarities.mean(axis=1)
            df['embedding_max_similarity'] = similarities.max(axis=1)
            df['embedding_min_similarity'] = similarities.min(axis=1)
        else:
            # Para test, calculamos similitud con los productos de train
            if self.embedding_matrix_train is not None:
                similarities = cosine_similarity(embeddings, self.embedding_matrix_train)
                df['embedding_avg_similarity'] = similarities.mean(axis=1)
                df['embedding_max_similarity'] = similarities.max(axis=1)
                df['embedding_min_similarity'] = similarities.min(axis=1)
            else:
                df['embedding_avg_similarity'] = 0
                df['embedding_max_similarity'] = 0
                df['embedding_min_similarity'] = 0
        
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
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('unknown'))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Manejar categorías no vistas
                        df[f'{col}_encoded'] = df[col].fillna('unknown').apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        df[f'{col}_encoded'] = -1
        
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
            agg_dict = {
                'weekly_demand': 'sum',  # Esta será nuestra variable objetivo
                'weekly_sales': 'sum'
            }
            
            # Mantener las features del producto (tomar el primer valor)
            feature_cols = [col for col in df.columns if col not in 
                          ['ID', 'weekly_demand', 'weekly_sales', 'num_week_iso', 'Production']]
            
            for col in feature_cols:
                agg_dict[col] = 'first'
            
            df_agg = df.groupby('ID', as_index=False).agg(agg_dict)
            
            # Renombrar para claridad
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
        
        # Features de embeddings (similitud de coseno)
        df = self.create_embedding_features(df, is_train=is_train)
        
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