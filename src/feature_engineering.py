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
        self.historical_stats = {}  # Guardar estadísticas históricas para test
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features temporales a partir de fechas"""
        df = df.copy()
        
        # Convertir phase_in y phase_out a datetime si no lo están
        if 'phase_in' in df.columns:
            df['phase_in'] = pd.to_datetime(df['phase_in'], errors='coerce')
            df['phase_out'] = pd.to_datetime(df['phase_out'], errors='coerce')
            
            # Features de phase_in
            df['phase_in_month'] = df['phase_in'].dt.month
            df['phase_in_quarter'] = df['phase_in'].dt.quarter
            df['phase_in_week'] = df['phase_in'].dt.isocalendar().week
            df['phase_in_dayofyear'] = df['phase_in'].dt.dayofyear
            
            # Features de phase_out
            df['phase_out_month'] = df['phase_out'].dt.month
            df['phase_out_quarter'] = df['phase_out'].dt.quarter
            
            # Season indicator (1: Spring-Summer, 2: Fall-Winter)
            df['season_type'] = df['phase_in_month'].apply(
                lambda x: 1 if x in [3, 4, 5, 6, 7, 8] else 2
            )
        
        return df
    
    def create_aggregated_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Crea features agregadas usando SOLO datos históricos (sin data leakage)"""
        df = df.copy()
        
        # Inicializar columnas
        df['family_mean_production_hist'] = np.nan
        df['family_std_production_hist'] = np.nan
        df['family_median_production_hist'] = np.nan
        df['category_mean_production_hist'] = np.nan
        df['category_median_production_hist'] = np.nan
        df['stores_mean_production_hist'] = np.nan
        
        if not is_train or TARGET not in df.columns or 'id_season' not in df.columns:
            # Para test, usar estadísticas guardadas
            if self.historical_stats:
                for col, stats_dict in self.historical_stats.items():
                    if col in df.columns:
                        df[f'{col}_mean_production_hist'] = df[col].map(stats_dict.get('mean', {}))
                        if 'std' in stats_dict:
                            df[f'{col}_std_production_hist'] = df[col].map(stats_dict['std'])
                        if 'median' in stats_dict:
                            df[f'{col}_median_production_hist'] = df[col].map(stats_dict['median'])
            
            # Rellenar NaN con medianas globales
            for col in df.columns:
                if '_hist' in col and df[col].isna().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            return df
        
        # Para train: calcular estadísticas usando solo temporadas anteriores
        seasons_sorted = sorted(df['id_season'].unique())
        
        # Para cada temporada, calcular stats de temporadas ANTERIORES
        for i, season in enumerate(seasons_sorted):
            # Máscara de la temporada actual
            season_mask = df['id_season'] == season
            
            if i == 0:
                # Primera temporada: usar valores globales (fallback)
                continue
            
            # Datos históricos (solo temporadas anteriores)
            hist_data = df[df['id_season'].isin(seasons_sorted[:i])]
            
            if len(hist_data) == 0:
                continue
            
            # --- Features por familia ---
            if 'family' in df.columns:
                family_stats = hist_data.groupby('family')[TARGET].agg(['mean', 'std', 'median']).to_dict()
                df.loc[season_mask, 'family_mean_production_hist'] = df.loc[season_mask, 'family'].map(family_stats['mean'])
                df.loc[season_mask, 'family_std_production_hist'] = df.loc[season_mask, 'family'].map(family_stats['std'])
                df.loc[season_mask, 'family_median_production_hist'] = df.loc[season_mask, 'family'].map(family_stats['median'])
            
            # --- Features por categoría ---
            if 'category' in df.columns:
                category_stats = hist_data.groupby('category')[TARGET].agg(['mean', 'median']).to_dict()
                df.loc[season_mask, 'category_mean_production_hist'] = df.loc[season_mask, 'category'].map(category_stats['mean'])
                df.loc[season_mask, 'category_median_production_hist'] = df.loc[season_mask, 'category'].map(category_stats['median'])
            
            # --- Features por número de tiendas ---
            if 'num_stores' in df.columns:
                stores_stats = hist_data.groupby('num_stores')[TARGET].mean().to_dict()
                df.loc[season_mask, 'stores_mean_production_hist'] = df.loc[season_mask, 'num_stores'].map(stores_stats)
        
        # Guardar estadísticas de TODAS las temporadas para aplicar al test
        if is_train:
            all_data = df.copy()
            
            if 'family' in all_data.columns:
                self.historical_stats['family'] = {
                    'mean': all_data.groupby('family')[TARGET].mean().to_dict(),
                    'std': all_data.groupby('family')[TARGET].std().to_dict(),
                    'median': all_data.groupby('family')[TARGET].median().to_dict()
                }
            
            if 'category' in all_data.columns:
                self.historical_stats['category'] = {
                    'mean': all_data.groupby('category')[TARGET].mean().to_dict(),
                    'median': all_data.groupby('category')[TARGET].median().to_dict()
                }
            
            if 'num_stores' in all_data.columns:
                self.historical_stats['num_stores'] = {
                    'mean': all_data.groupby('num_stores')[TARGET].mean().to_dict()
                }
        
        # Rellenar NaN con medianas de columnas
        for col in ['family_mean_production_hist', 'family_std_production_hist', 'family_median_production_hist',
                    'category_mean_production_hist', 'category_median_production_hist', 'stores_mean_production_hist']:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de interacción"""
        df = df.copy()
        
        if 'num_stores' in df.columns and 'num_sizes' in df.columns:
            # Capacidad total (tiendas * tamaños)
            df['total_capacity'] = df['num_stores'] * df['num_sizes']
        
        if 'num_stores' in df.columns and 'price' in df.columns:
            # Potencial de ingresos
            df['revenue_potential'] = df['num_stores'] * df['price']
        
        if 'price' in df.columns and 'num_sizes' in df.columns:
            # Precio promedio por tamaño
            df['price_per_size'] = df['price'] / (df['num_sizes'] + 1)
        
        if 'life_cycle_length' in df.columns and 'num_stores' in df.columns:
            # Exposición total (semanas * tiendas)
            df['total_exposure'] = df['life_cycle_length'] * df['num_stores']
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Crea features de lag basadas en productos similares (solo históricos)"""
        df = df.copy()
        
        # Inicializar columnas de lag
        df['family_lag1_production'] = np.nan
        df['family_lag2_production'] = np.nan
        
        if not is_train:
            # Para test, no podemos calcular lags sin el target
            # Rellenar con mediana de train (si se guardó)
            return df
        
        # Ordenar por familia y temporada
        if 'family' in df.columns and 'id_season' in df.columns and TARGET in df.columns:
            df = df.sort_values(['family', 'id_season'])
            
            # Lag de producción por familia (shift ya garantiza uso de datos anteriores)
            df['family_lag1_production'] = df.groupby('family')[TARGET].shift(1)
            df['family_lag2_production'] = df.groupby('family')[TARGET].shift(2)
            
            # Rellenar NaN con medianas
            df['family_lag1_production'].fillna(df['family_lag1_production'].median(), inplace=True)
            df['family_lag2_production'].fillna(df['family_lag2_production'].median(), inplace=True)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                    categorical_cols: List[str],
                                    is_train: bool = True) -> pd.DataFrame:
        """Codifica features categóricas usando target encoding o frequency encoding"""
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
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
        df = self.create_aggregated_features(df, is_train=True)
        df = self.create_interaction_features(df)
        df = self.create_lag_features(df, is_train=True)
        df = self.process_image_embeddings(df)
        df = self.encode_categorical_features(df, categorical_cols, is_train=True)
        
        return df
    
    def transform(self, df: pd.DataFrame, 
                  categorical_cols: List[str]) -> pd.DataFrame:
        """Aplica transformaciones en datos de test (sin data leakage)"""
        df = self.create_temporal_features(df)
        df = self.create_aggregated_features(df, is_train=False)
        df = self.create_interaction_features(df)
        df = self.create_lag_features(df, is_train=False)
        df = self.process_image_embeddings(df)
        df = self.encode_categorical_features(df, categorical_cols, is_train=False)
        
        return df


# Variable target
TARGET = 'Production'

