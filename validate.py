import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append('src')

from feature_engineering import FeatureProcessor
from model import DemandPredictor, calculate_metrics


def validate_model(train_csv_path: str, test_size: float = 0.2, random_state: int = 42):
    
    print("="*80)
    print("VALIDACIÓN DEL MODELO")
    print("="*80)
    
    # Cargar datos
    print("\n1. Cargando datos...")
    df = pd.read_csv(train_csv_path, delimiter=';', decimal=',')
    print(f"   Dataset: {df.shape}")
    
    # Split de IDs
    print("\n2. Split a nivel producto...")
    unique_ids = df['ID'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    
    df_train = df[df['ID'].isin(train_ids)].copy()
    df_val = df[df['ID'].isin(val_ids)].copy()
    
    print(f"   Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")
    
    # Procesar features
    print("\n3. Procesando features...")
    processor = FeatureProcessor()
    
    df_train_processed = processor.process_features(df_train, is_train=True, mode='train')
    df_val_processed = processor.process_features(df_val, is_train=False, mode='train')
    
    # Preparar datos
    feature_cols = processor.get_feature_columns(df_train_processed)
    
    X_train = df_train_processed[feature_cols]
    y_train = df_train_processed['demand_total']
    
    X_val = df_val_processed[feature_cols]
    y_val = df_val_processed['demand_total']
    
    print(f"   Features: {len(feature_cols)}")
    
    # Entrenar
    print("\n4. Entrenando LightGBM...")
    model = DemandPredictor()
    model.train(X_train, y_train, X_val, y_val)
    
    # Predecir
    y_pred = model.predict(X_val)
    
    # Evaluar
    metrics = calculate_metrics(y_val.values, y_pred)
    
    print("\n" + "="*80)
    print("RESULTADOS")
    print("="*80)
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"Ventas Perdidas (avg): {metrics['avg_lost_sales_per_product']:.2f}")
    print(f"Exceso Stock (avg): {metrics['avg_excess_stock_per_product']:.2f}")
    print(f"\nSCORE: {metrics['custom_score']:.2f} / 100")
    print("="*80)
    
    # Guardar
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    results_df = pd.DataFrame({
        'ID': df_val_processed['ID'],
        'demand_total_real': y_val.values,
        'demand_total_predicted': y_pred
    })
    
    results_df.to_csv('outputs/validation_results.csv', index=False)
    model.save_model('models/lgbm_model_validation.pkl')
    
    print("\nGuardado en outputs/ y models/")
    
    return model, metrics, results_df


if __name__ == "__main__":
    model, metrics, results = validate_model('data/train.csv', test_size=0.2, random_state=42)