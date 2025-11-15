"""
Script principal de entrenamiento del modelo
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Añadir src al path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_engineering import FeatureEngineer
from model import DemandPredictor, optimize_hyperparameters
import config


def load_data():
    """Carga los datos de entrenamiento y test"""
    print("📂 Cargando datos...")
    
    # Los archivos usan punto y coma (;) como delimitador
    train_df = pd.read_csv(config.TRAIN_FILE, sep=';', low_memory=False)
    test_df = pd.read_csv(config.TEST_FILE, sep=';', low_memory=False)
    
    # Sample submission usa comas
    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
    
    print(f"   ✓ Train: {train_df.shape}")
    print(f"   ✓ Test: {test_df.shape}")
    
    return train_df, test_df, sample_submission


def prepare_features(train_df, test_df, optimize: bool = False):
    """Prepara las features para el modelo"""
    print("\n🔧 Ingeniería de features...")
    
    # Crear ingeniero de features
    fe = FeatureEngineer()
    
    # Aplicar transformaciones
    print("   - Creando features temporales...")
    train_processed = fe.fit_transform(train_df, config.CATEGORICAL_FEATURES)
    
    print("   - Aplicando transformaciones a test...")
    test_processed = fe.transform(test_df, config.CATEGORICAL_FEATURES)
    
    # Agregar datos de weekly para crear features agregadas
    # Nota: El dataset tiene weekly_sales y weekly_demand que necesitamos agregar
    print("   - Agregando datos por producto...")
    
    # Identificar columnas disponibles
    available_cols = train_processed.columns.tolist()
    
    # Agregar ventas/demanda semanales por producto si existen
    if 'weekly_sales' in train_processed.columns and 'ID' in train_processed.columns:
        weekly_agg = train_processed.groupby('ID').agg({
            'weekly_sales': ['sum', 'mean', 'max', 'std'],
            'weekly_demand': ['sum', 'mean', 'max', 'std']
        }).reset_index()
        
        # Aplanar nombres de columnas
        weekly_agg.columns = ['ID'] + [f'{col[0]}_{col[1]}' for col in weekly_agg.columns[1:]]
        
        # Merge con train
        train_processed = train_processed.merge(weekly_agg, on='ID', how='left')
        
        # Para test, usar estadísticas globales
        for col in weekly_agg.columns:
            if col != 'ID' and col not in test_processed.columns:
                test_processed[col] = train_processed[col].median()
    
    # Seleccionar features para el modelo
    # Excluir columnas que no deben usarse
    exclude_cols = config.EXCLUDE_COLS + ['phase_in', 'phase_out']
    
    # Obtener todas las columnas numéricas
    feature_cols = [col for col in train_processed.columns 
                   if col not in exclude_cols and 
                   train_processed[col].dtype in ['int64', 'float64']]
    
    # Asegurarse de que test tiene las mismas columnas
    missing_in_test = set(feature_cols) - set(test_processed.columns)
    if missing_in_test:
        print(f"   ⚠️  Columnas faltantes en test: {missing_in_test}")
        for col in missing_in_test:
            test_processed[col] = 0
    
    # Filtrar NaN
    train_processed[feature_cols] = train_processed[feature_cols].fillna(0)
    test_processed[feature_cols] = test_processed[feature_cols].fillna(0)
    
    # Preparar datos
    X_train = train_processed[feature_cols]
    y_train = train_processed[config.TARGET]
    X_test = test_processed[feature_cols]
    
    print(f"   ✓ Features creadas: {len(feature_cols)}")
    print(f"   ✓ X_train shape: {X_train.shape}")
    print(f"   ✓ X_test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, test_processed['ID'], feature_cols, fe


def train_model(X_train, y_train, optimize_params: bool = False):
    """Entrena el modelo"""
    print("\n🤖 Entrenando modelo XGBoost...")
    
    # Optimizar hiperparámetros si se solicita
    if optimize_params:
        print("   🔍 Optimizando hiperparámetros con Optuna...")
        best_params = optimize_hyperparameters(
            X_train, y_train,
            n_trials=config.OPTUNA_N_TRIALS,
            timeout=config.OPTUNA_TIMEOUT,
            random_state=config.RANDOM_STATE
        )
        params = {**config.XGBOOST_PARAMS, **best_params}
    else:
        params = config.XGBOOST_PARAMS
    
    # Crear y entrenar modelo
    predictor = DemandPredictor(params=params, random_state=config.RANDOM_STATE)
    
    print("\n   📊 Validación cruzada...")
    cv_results = predictor.cross_validate(X_train, y_train, 
                                         n_splits=config.N_SPLITS,
                                         verbose=True)
    
    # Entrenar en todo el dataset
    print("\n   🎯 Entrenamiento final en todos los datos...")
    predictor.train(X_train, y_train, verbose=True)
    
    # Guardar modelo
    print("\n   💾 Guardando modelo...")
    predictor.save(config.MODELS_DIR)
    
    # Mostrar feature importance
    print("\n   📈 Top 20 Features más importantes:")
    if predictor.feature_importance is not None:
        print(predictor.feature_importance.head(20).to_string(index=False))
    
    return predictor, cv_results


def make_predictions(predictor, X_test, test_ids, sample_submission):
    """Genera predicciones y crea archivo de submission"""
    print("\n🔮 Generando predicciones...")
    
    predictions = predictor.predict(X_test)
    
    # Crear submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'Production': predictions
    })
    
    # Asegurarse de que el formato sea correcto
    submission['Production'] = submission['Production'].round(0).astype(int)
    
    # Estadísticas
    print(f"\n   📊 Estadísticas de predicciones:")
    print(f"      - Media: {predictions.mean():.2f}")
    print(f"      - Mediana: {np.median(predictions):.2f}")
    print(f"      - Min: {predictions.min():.2f}")
    print(f"      - Max: {predictions.max():.2f}")
    print(f"      - Std: {predictions.std():.2f}")
    
    # Guardar submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = config.SUBMISSIONS_DIR / f"submission_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"\n   ✅ Submission guardado en: {submission_path}")
    
    return submission


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrena modelo de predicción de demanda')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimizar hiperparámetros con Optuna')
    parser.add_argument('--no-cv', action='store_true',
                       help='Saltar validación cruzada (más rápido)')
    args = parser.parse_args()
    
    print("="*60)
    print("🥭 MANGO - Predicción de Demanda")
    print("="*60)
    
    # Crear directorios si no existen
    for directory in [config.MODELS_DIR, config.SUBMISSIONS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Cargar datos
    train_df, test_df, sample_submission = load_data()
    
    # Preparar features
    X_train, y_train, X_test, test_ids, feature_cols, fe = prepare_features(
        train_df, test_df, optimize=args.optimize
    )
    
    # Entrenar modelo
    predictor, cv_results = train_model(X_train, y_train, 
                                       optimize_params=args.optimize)
    
    # Generar predicciones
    submission = make_predictions(predictor, X_test, test_ids, sample_submission)
    
    print("\n" + "="*60)
    print("✨ ¡Proceso completado exitosamente!")
    print("="*60)
    print(f"\n📋 Resumen de resultados:")
    print(f"   - Custom Score (CV): {cv_results['custom_score']['mean']:.2f} ± {cv_results['custom_score']['std']:.2f}")
    print(f"   - VAR (CV): {cv_results['var']['mean']:.4f} ± {cv_results['var']['std']:.4f}")
    print(f"   - RMSE (CV): {cv_results['rmse']['mean']:.4f} ± {cv_results['rmse']['std']:.4f}")
    print(f"\n🎯 Siguiente paso: Sube el archivo de submission a la plataforma!")
    print("="*60)


if __name__ == "__main__":
    main()

