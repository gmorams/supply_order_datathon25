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
    df_train = pd.read_csv(train_csv_path, delimiter=';', decimal=',')
    df_test = pd.read_csv(test_csv_path, delimiter=';', decimal=',')
    
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
    
    # Entrenar con K-Fold Cross-Validation
    print("\n3. Entrenando modelo con K-Fold (5 folds)...")
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X_train))
    test_preds_folds = []
    models = []  # Guardar todos los modelos
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"   Fold {fold+1}/5...")
        
        X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
        y_trn, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]
        
        fold_model = DemandPredictor()
        fold_model.train(X_trn, y_trn)
        models.append(fold_model)
        
        # OOF (out-of-fold predictions para evaluación)
        oof_preds[val_idx] = fold_model.predict(X_val)
        
        # Predicciones de test para este fold
        test_pred_fold = fold_model.predict(X_test)
        test_preds_folds.append(test_pred_fold)
    
    # Promedio de las predicciones de los 5 folds
    print("\n4. Generando predicciones finales (promedio de folds)...")
    predictions = np.mean(test_preds_folds, axis=0)
    predictions = np.maximum(predictions, 0)
    
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
    
    # Guardar todos los modelos de los folds
    for i, fold_model in enumerate(models):
        fold_model.save_model(f'models/lgbm_model_fold_{i+1}.pkl')
    
    print("\n" + "="*80)
    print("SUBMISSION GENERADO")
    print("="*80)
    print(f"Archivo: {output_path}")
    print(f"Predicciones: {len(submission)}")
    print(f"Producción total: {submission['Production'].sum():,}")
    print(f"Modelos guardados: {len(models)} folds")
    
    print(f"\nPrimeras líneas del submission:")
    print(submission.head(10).to_string(index=False))
    
    return submission, models


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