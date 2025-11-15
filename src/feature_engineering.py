"""
Feature Engineering SIMPLE - Solo lo básico
"""
import pandas as pd
import numpy as np
from typing import List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Clase simple para preparar features para el modelo"""
    
    def __init__(self):
        self.label_encoders = {}
        
    def encode_categorical(self, df: pd.DataFrame, 
                          categorical_cols: List[str],
                          is_train: bool = True) -> pd.DataFrame:
        """
        Encoding simple de variables categóricas
        Convierte texto a números
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            if is_train:
                # Crear mapeo de categoría -> número
                unique_values = df[col].dropna().unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                self.label_encoders[col] = encoding_map
            
            # Aplicar encoding
            if col in self.label_encoders:
                df[col + '_encoded'] = df[col].map(self.label_encoders[col])
                # Rellenar valores desconocidos con -1
                df[col + '_encoded'] = df[col + '_encoded'].fillna(-1)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rellena valores nulos de forma simple
        """
        df = df.copy()
        
        # Rellenar numéricos con mediana
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Rellenar categóricos con 'missing'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna('missing')
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, 
                     categorical_cols: List[str]) -> pd.DataFrame:
        """
        Prepara datos de TRAIN
        """
        print("   • Rellenando valores nulos...")
        df = self.handle_missing_values(df)
        
        print("   • Codificando variables categóricas...")
        df = self.encode_categorical(df, categorical_cols, is_train=True)
        
        return df
    
    def transform(self, df: pd.DataFrame, 
                  categorical_cols: List[str]) -> pd.DataFrame:
        """
        Prepara datos de TEST (usa encodings del train)
        """
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df, categorical_cols, is_train=False)
        
        return df


# Variable target
TARGET = 'Production'
