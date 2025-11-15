"""
Módulo de ingeniería de features para el desafío de predicción de demanda
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Clase para crear features adicionales para el modelo"""
    
    def __init__(self):
        self.encoding_maps = {}
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features temporales a partir de fechas"""
        df = df.copy()
        
        # Convertir phase_in y phase_out a datetime si no lo están
        if 'phase_in' in df.columns:
            df['phase_in'] = pd.to_datetime(df['phase_in'], errors='coerce')
            df['phase_out'] = pd.to_datetime(df['phase_out'], errors='coerce')
            
            # Features básicas de phase_in (útiles para capturar estacionalidad)
            df['phase_in_month'] = df['phase_in'].dt.month
            df['phase_in_quarter'] = df['phase_in'].dt.quarter
            
            # Season indicator (Spring-Summer vs Fall-Winter) - MUY importante en moda
            df['is_spring_summer'] = df['phase_in_month'].apply(
                lambda x: 1 if x in [3, 4, 5, 6, 7, 8] else 0
            )
        
        return df
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features agregadas por grupo.
        
        ACLARACIÓN: Estas NO son features numéricas directas.
        Son estadísticas calculadas AGRUPANDO por categorías.
        Por ejemplo: "promedio de producción para productos de la familia Dresses"
        """
        df = df.copy()
        
        # Agregar SOLO media por familia (lo más importante)
        if 'family' in df.columns and TARGET in df.columns:
            family_stats = df.groupby('family')[TARGET].agg([
                ('family_mean_production', 'mean')
            ]).reset_index()
            df = df.merge(family_stats, on='family', how='left')
        
        # Media por categoría
        if 'category' in df.columns and TARGET in df.columns:
            category_stats = df.groupby('category')[TARGET].agg([
                ('category_mean_production', 'mean')
            ]).reset_index()
            df = df.merge(category_stats, on='category', how='left')
        
        # Media por número de tiendas (productos con misma distribución)
        if 'num_stores' in df.columns and TARGET in df.columns:
            stores_stats = df.groupby('num_stores')[TARGET].agg([
                ('stores_mean_production', 'mean')
            ]).reset_index()
            df = df.merge(stores_stats, on='num_stores', how='left')
        
        # Media por temporada
        if 'id_season' in df.columns and TARGET in df.columns:
            season_stats = df.groupby('id_season')[TARGET].agg([
                ('season_mean_production', 'mean')
            ]).reset_index()
            df = df.merge(season_stats, on='id_season', how='left')
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de interacción (COMENTADAS - no usar por ahora)
        Solo descomenta si los resultados base no son buenos
        """
        df = df.copy()
        
        # COMENTADO: Capacidad total
        # if 'num_stores' in df.columns and 'num_sizes' in df.columns:
        #     df['total_capacity'] = df['num_stores'] * df['num_sizes']
        
        # COMENTADO: Potencial de ingresos
        # if 'num_stores' in df.columns and 'price' in df.columns:
        #     df['revenue_potential'] = df['num_stores'] * df['price']
        
        # COMENTADO: Precio por tamaño
        # if 'price' in df.columns and 'num_sizes' in df.columns:
        #     df['price_per_size'] = df['price'] / (df['num_sizes'] + 1)
        
        # COMENTADO: Exposición total
        # if 'life_cycle_length' in df.columns and 'num_stores' in df.columns:
        #     df['total_exposure'] = df['life_cycle_length'] * df['num_stores']
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features de lag (COMENTADAS - no usar por ahora)
        Los lags pueden causar data leakage o no funcionar bien con productos nuevos
        """
        df = df.copy()
        
        # COMENTADO: Lag por familia
        # if 'family' in df.columns and 'id_season' in df.columns:
        #     df = df.sort_values(['family', 'id_season'])
        #     if TARGET in df.columns:
        #         df['family_lag1_production'] = df.groupby('family')[TARGET].shift(1)
        #         df['family_lag2_production'] = df.groupby('family')[TARGET].shift(2)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                    categorical_cols: List[str],
                                    is_train: bool = True) -> pd.DataFrame:
        """
        Codifica features categóricas.
        - Target encoding para features con muchas categorías
        - One-hot encoding para features con pocas categorías (<=5)
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            # Contar categorías únicas
            n_unique = df[col].nunique()
            
            # One-hot encoding para variables con pocas categorías
            if n_unique <= 5 and n_unique > 1:
                # One-hot encoding (crear columnas dummy)
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df, dummies], axis=1)
            
            # Target encoding para variables con muchas categorías
            else:
                if is_train:
                    # Target encoding (solo en train)
                    if TARGET in df.columns:
                        encoding_map = df.groupby(col)[TARGET].mean().to_dict()
                        self.encoding_maps[col] = encoding_map
                    else:
                        # Frequency encoding
                        encoding_map = df[col].value_counts(normalize=True).to_dict()
                        self.encoding_maps[col] = encoding_map
                
                # Aplicar encoding
                if col in self.encoding_maps:
                    df[f'{col}_encoded'] = df[col].map(self.encoding_maps[col])
                    # Rellenar valores no vistos con la mediana
                    df[f'{col}_encoded'].fillna(df[f'{col}_encoded'].median(), inplace=True)
        
        return df
    
    def process_image_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa los embeddings de imagen"""
        df = df.copy()
        
        # Los embeddings están en una sola columna como string separado por comas
        if 'image_embedding' in df.columns:
            try:
                # Intentar convertir el string de embeddings a lista de números
                def parse_embedding(emb_str):
                    if pd.isna(emb_str) or emb_str == '':
                        return [0] * 10  # Vector vacío de tamaño fijo
                    try:
                        values = [float(x.strip()) for x in str(emb_str).split(',')]
                        return values
                    except:
                        return [0] * 10
                
                # Parsear embeddings
                embeddings = df['image_embedding'].apply(parse_embedding)
                
                # Crear features estadísticas
                df['embedding_mean'] = embeddings.apply(lambda x: np.mean(x) if len(x) > 0 else 0)
                df['embedding_std'] = embeddings.apply(lambda x: np.std(x) if len(x) > 0 else 0)
                df['embedding_max'] = embeddings.apply(lambda x: np.max(x) if len(x) > 0 else 0)
                df['embedding_min'] = embeddings.apply(lambda x: np.min(x) if len(x) > 0 else 0)
                df['embedding_len'] = embeddings.apply(len)
            except Exception as e:
                print(f"   ⚠️  Error processing image embeddings: {e}")
                # Si falla, simplemente eliminar la columna
                pass
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, 
                     categorical_cols: List[str]) -> pd.DataFrame:
        """Aplica todas las transformaciones en datos de entrenamiento"""
        df = self.create_temporal_features(df)
        df = self.create_aggregated_features(df)
        # df = self.create_interaction_features(df)  # COMENTADO
        # df = self.create_lag_features(df)  # COMENTADO
        df = self.process_image_embeddings(df)
        df = self.encode_categorical_features(df, categorical_cols, is_train=True)
        
        return df
    
    def transform(self, df: pd.DataFrame, 
                  categorical_cols: List[str]) -> pd.DataFrame:
        """Aplica transformaciones en datos de test"""
        df = self.create_temporal_features(df)
        # df = self.create_interaction_features(df)  # COMENTADO
        df = self.process_image_embeddings(df)
        df = self.encode_categorical_features(df, categorical_cols, is_train=False)
        
        return df


# Variable target
TARGET = 'Production'