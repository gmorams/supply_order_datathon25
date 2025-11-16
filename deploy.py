import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.append('src')

from feature_engineering import FeatureProcessor
from model import DemandPredictor


def generate_submission(train_csv_path: str, test_csv_path: str):
    
    print("="*80)
    print("GENERACIÓN DE SUBMISSION")
    print("="*80)
    
    # Cargar datos
    print("\n1. Cargando datos...")
    df_train = pd.read_csv(train_csv_path, delimiter=';')
    df_test = pd.read_csv(test_csv_path, delimiter=';')
    
    print(f"   Train: {df_train.shape}")
    print(f"   Test: {df_test.shape}")
    
    # Procesar features
    print("\n2. Procesando features...")
    processor = FeatureProcessor()
    df_train_processed = processor.process_features(df_train, is_train=True, mode='train')
    df_test_processed = processor.process_features(df_test, is_train=False, mode='test')
    
    # Preparar datos
    feature_cols = processor.get_feature_columns(df_train_processed)
    
    X_train = df_train_processed[feature_cols]
    y_train = df_train_processed['demand_total']
    X_test = df_test_processed[feature_cols]
    
    print(f"   Features: {len(feature_cols)}")
    
    # Entrenar
    print("\n3. Entrenando modelo final...")
    model = DemandPredictor()
    model.train(X_train, y_train)
    
    # Predecir
    print("\n4. Generando predicciones...")
    predictions = model.predict(X_test)
    
    # Crear submission
    submission = pd.DataFrame({
        'ID': df_test['ID'],
        'Production': predictions.astype(int)
    })
    
    # Guardar
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'outputs/submission_{timestamp}.csv'
    
    submission.to_csv(output_path, index=False)
    model.save_model('models/lgbm_model_final.pkl')
    
    print("\n" + "="*80)
    print("SUBMISSION GENERADO")
    print("="*80)
    print(f"Archivo: {output_path}")
    print(f"Predicciones: {len(submission)}")
    print(f"Producción total: {submission['Production'].sum():,}")
    
    print(f"\nPrimeras líneas del submission:")
    print(submission.head(10).to_string(index=False))
    
    return submission, model


if __name__ == "__main__":
    submission, model = generate_submission('data/train.csv', 'data/test.csv')
'''

**Formato del submission será exactamente:**

ID,Production
90,1500
16,1500
65,1500
138,1500
...
'''