"""
Módulo de ingeniería de features para el desafío de predicción de demanda
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Clase para crear features adicionales para el modelo"""
    
    def __init__(self):
        self.encoding_maps = {}
        self.historical_embeddings = None  # Guardar embeddings históricos
        self.scaler = StandardScaler()
        
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
    
    def create_weekly_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features que capturan el comportamiento semanal.
        Las agrega directamente sin duplicar filas.
        """
        df = df.copy()
        
        # Solo procesar si tenemos datos semanales
        if 'weekly_sales' not in df.columns or 'weekly_demand' not in df.columns:
            return df
        
        print(f"      → Calculando patrones semanales de {df['ID'].nunique()} productos...")
        
        # Agrupar por ID y calcular TODO de una vez
        weekly_agg = df.groupby('ID').agg({
            'weekly_sales': [
                ('sales_total', 'sum'),
                ('sales_mean', 'mean'),
                ('sales_max', 'max'),
                ('sales_std', 'std')
            ],
            'weekly_demand': [
                ('demand_total_calc', 'sum'),  # Verificación
                ('demand_mean', 'mean'),
                ('demand_max', 'max'),
                ('demand_std', 'std')
            ],
            'num_week_iso': [('num_weeks', 'count')]
        })
        
        # Aplanar columnas
        weekly_agg.columns = [
            'sales_total', 'sales_mean', 'sales_max', 'sales_std',
            'demand_total_calc', 'demand_mean', 'demand_max', 'demand_std',
            'num_weeks'
        ]
        weekly_agg = weekly_agg.reset_index()
        
        # Calcular tendencia (más eficiente)
        def calc_trend_by_id(group):
            x = np.arange(len(group))
            y = group['weekly_sales'].values
            if len(y) < 2:
                return 0
            try:
                return np.polyfit(x, y, 1)[0]  # Pendiente
            except:
                return 0
        
        trends = df.groupby('ID').apply(calc_trend_by_id).rename('sales_trend')
        weekly_agg = weekly_agg.merge(trends, left_on='ID', right_index=True)
        
        # Ratios
        weekly_agg['sales_to_demand_ratio'] = (
            weekly_agg['sales_total'] / (weekly_agg['demand_total_calc'] + 1)
        )
        weekly_agg['is_growing'] = (weekly_agg['sales_trend'] > 0).astype(int)
        weekly_agg['is_stable'] = (
            (weekly_agg['sales_std'] / (weekly_agg['sales_mean'] + 1)) < 0.3
        ).astype(int)
        
        # Merge con df original (esto NO duplica, solo añade columnas)
        df = df.merge(weekly_agg.drop('demand_total_calc', axis=1), on='ID', how='left')
        
        return df
    
    def _calculate_trend(self, series):
        """Calcula la tendencia de una serie temporal (pendiente de regresión lineal)"""
        if len(series) < 2:
            return 0
        
        x = np.arange(len(series))
        y = series.values
        
        # Regresión lineal simple: y = mx + b
        try:
            m = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
            return m
        except:
            return 0
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features agregadas por grupo.
        Estadísticas de producción histórica por categorías.
        """
        df = df.copy()
        
        # Media por familia
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
        
        # Media por número de tiendas
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
    
    def process_image_embeddings(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Procesa embeddings de imagen calculando similitud con productos históricos exitosos.
        
        Estrategia:
        - En TRAIN: Guardar embeddings de productos con alta producción (exitosos)
        - En TEST: Calcular cosine similarity con esos productos exitosos
        """
        df = df.copy()
        
        if 'image_embedding' not in df.columns:
            return df
        
        try:
            # Parsear embeddings de string a array
            def parse_embedding(emb_str):
                if pd.isna(emb_str) or emb_str == '':
                    return None
                try:
                    values = np.array([float(x.strip()) for x in str(emb_str).split(',')])
                    return values
                except:
                    return None
            
            embeddings_list = df['image_embedding'].apply(parse_embedding)
            
            # Filtrar IDs con embeddings válidos
            valid_mask = embeddings_list.notna()
            valid_indices = df[valid_mask].index
            
            if len(valid_indices) == 0:
                print("   ⚠️  No hay embeddings válidos")
                return df
            
            # Convertir a matriz numpy
            embeddings_matrix = np.vstack(embeddings_list[valid_mask].values)
            
            if is_train:
                # Guardar embeddings históricos de productos exitosos
                # Definir "exitoso" como productos con Production > percentil 75
                if TARGET in df.columns:
                    threshold = df[TARGET].quantile(0.75)
                    successful_mask = df[TARGET] > threshold
                    successful_indices = df[valid_mask & successful_mask].index
                    
                    if len(successful_indices) > 0:
                        self.historical_embeddings = embeddings_matrix[
                            df[valid_mask].index.isin(successful_indices)
                        ]
                        print(f"   ✓ Guardados {len(self.historical_embeddings)} embeddings de productos exitosos")
                    else:
                        # Si no hay productos exitosos, usar todos
                        self.historical_embeddings = embeddings_matrix
                else:
                    self.historical_embeddings = embeddings_matrix
                
                # Calcular similitud consigo mismo (para train)
                similarities = cosine_similarity(embeddings_matrix, self.historical_embeddings)
                
            else:
                # TEST: Calcular similitud con embeddings históricos
                if self.historical_embeddings is None:
                    print("   ⚠️  No hay embeddings históricos guardados")
                    # Features por defecto
                    df['embedding_max_similarity'] = 0.5
                    df['embedding_mean_similarity'] = 0.5
                    df['embedding_top3_similarity'] = 0.5
                    return df
                
                similarities = cosine_similarity(embeddings_matrix, self.historical_embeddings)
            
            # Crear features de similitud
            similarity_features = pd.DataFrame(index=valid_indices)
            
            # Máxima similitud (producto histórico más parecido)
            similarity_features['embedding_max_similarity'] = similarities.max(axis=1)
            
            # Similitud promedio
            similarity_features['embedding_mean_similarity'] = similarities.mean(axis=1)
            
            # Top-3 similitud promedio
            top3_similarities = np.sort(similarities, axis=1)[:, -3:]
            similarity_features['embedding_top3_similarity'] = top3_similarities.mean(axis=1)
            
            # Merge con df original
            df = df.merge(similarity_features, left_index=True, right_index=True, how='left')
            
            # Rellenar NaN (productos sin embedding) con valores neutros
            df['embedding_max_similarity'].fillna(0.5, inplace=True)
            df['embedding_mean_similarity'].fillna(0.5, inplace=True)
            df['embedding_top3_similarity'].fillna(0.5, inplace=True)
            
            print(f"   ✓ Features de similitud creadas basadas en {len(self.historical_embeddings)} embeddings históricos")
            
        except Exception as e:
            print(f"   ⚠️  Error procesando embeddings: {e}")
            # Features por defecto en caso de error
            df['embedding_max_similarity'] = 0.5
            df['embedding_mean_similarity'] = 0.5
            df['embedding_top3_similarity'] = 0.5
        
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
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df, dummies], axis=1)
            
            # Target encoding para variables con muchas categorías
            else:
                if is_train:
                    if TARGET in df.columns:
                        encoding_map = df.groupby(col)[TARGET].mean().to_dict()
                        self.encoding_maps[col] = encoding_map
                    else:
                        encoding_map = df[col].value_counts(normalize=True).to_dict()
                        self.encoding_maps[col] = encoding_map
                
                if col in self.encoding_maps:
                    df[f'{col}_encoded'] = df[col].map(self.encoding_maps[col])
                    df[f'{col}_encoded'].fillna(df[f'{col}_encoded'].median(), inplace=True)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, 
                     categorical_cols: List[str]) -> pd.DataFrame:
        """Aplica todas las transformaciones en datos de entrenamiento"""
        print("   - Features temporales...")
        df = self.create_temporal_features(df)
        
        print("   - Features de comportamiento semanal...")
        df = self.create_weekly_behavior_features(df)
        
        print("   - Features agregadas...")
        df = self.create_aggregated_features(df)
        
        print("   - Procesando image embeddings (train)...")
        df = self.process_image_embeddings(df, is_train=True)
        
        print("   - Encoding categóricas...")
        df = self.encode_categorical_features(df, categorical_cols, is_train=True)
        
        return df
    
    def transform(self, df: pd.DataFrame, 
                  categorical_cols: List[str]) -> pd.DataFrame:
        """Aplica transformaciones en datos de test"""
        print("   - Features temporales...")
        df = self.create_temporal_features(df)
        
        print("   - Procesando image embeddings (test)...")
        df = self.process_image_embeddings(df, is_train=False)
        
        print("   - Encoding categóricas...")
        df = self.encode_categorical_features(df, categorical_cols, is_train=False)
        
        return df


# Variable target
TARGET = 'demand_total'