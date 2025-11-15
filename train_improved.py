"""
Script mejorado de entrenamiento con estrategias anti-overfitting
y mejores features para aumentar el score en Kaggle
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'src'))

from feature_engineering import FeatureEngineer
from model import DemandPredictor
import config


def load_data():
    """Carga los datos de entrenamiento y test"""
    print("📂 Cargando datos...")
    
    train_df = pd.read_csv(config.TRAIN_FILE, sep=';', low_memory=False)
    test_df = pd.read_csv(config.TEST_FILE, sep=';', low_memory=False)
    sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
    
    print(f"   ✓ Train: {train_df.shape}")
    print(f"   ✓ Test: {test_df.shape}")
    
    return train_df, test_df, sample_submission


def create_advanced_features(train_df, test_df):
    """Crea features avanzadas para mejorar el modelo"""
    print("\n🔧 Creando features avanzadas...")
    
    # Agregar por ID para tener estadísticas por producto
    print("   - Agregando por producto (ID)...")
    if 'ID' in train_df.columns:
        train_agg = train_df.groupby('ID').agg({
            'weekly_sales': ['sum', 'mean', 'std', 'max', 'min', 'median'],
            'weekly_demand': ['sum', 'mean', 'std', 'max', 'min'],
            'Production': ['mean']
        }).reset_index()
        
        train_agg.columns = ['ID'] + [f'{col[0]}_{col[1]}' for col in train_agg.columns[1:]]
        
        # Merge con train
        train_df = train_df.merge(train_agg, on='ID', how='left', suffixes=('', '_agg'))
        
        # Para test, usar medias globales
        for col in train_agg.columns:
            if col != 'ID' and col not in test_df.columns:
                test_df[col] = train_agg[col].median()
    
    # Ratios importantes
    print("   - Creando ratios...")
    for df in [train_df, test_df]:
        if 'weekly_demand_sum' in df.columns and 'weekly_sales_sum' in df.columns:
            df['sales_demand_ratio'] = df['weekly_sales_sum'] / (df['weekly_demand_sum'] + 1)
            df['demand_fulfillment'] = df['weekly_sales_sum'] / (df['weekly_demand_sum'] + 1)
        
        if 'num_stores' in df.columns and 'price' in df.columns:
            df['price_x_stores'] = df['price'] * df['num_stores']
            df['avg_sales_per_store'] = df.get('weekly_sales_sum', 0) / (df['num_stores'] + 1)
        
        if 'life_cycle_length' in df.columns:
            df['sales_per_week'] = df.get('weekly_sales_sum', 0) / (df['life_cycle_length'] + 1)
            df['demand_per_week'] = df.get('weekly_demand_sum', 0) / (df['life_cycle_length'] + 1)
    
    print("   ✓ Features avanzadas creadas")
    return train_df, test_df


def prepare_features_improved(train_df, test_df):
    """Prepara features con enfoque en generalización"""
    print("\n🔧 Ingeniería de features mejorada...")
    
    # Crear features avanzadas primero
    train_df, test_df = create_advanced_features(train_df, test_df)
    
    # Feature engineering estándar
    fe = FeatureEngineer()
    train_processed = fe.fit_transform(train_df, config.CATEGORICAL_FEATURES)
    test_processed = fe.transform(test_df, config.CATEGORICAL_FEATURES)
    
    # Seleccionar features (excluyendo las problemáticas)
    exclude_cols = config.EXCLUDE_COLS + ['phase_in', 'phase_out', 'image_embedding', 'color_rgb']
    
    feature_cols = [col for col in train_processed.columns 
                   if col not in exclude_cols and 
                   train_processed[col].dtype in ['int64', 'float64']]
    
    # Rellenar NaN
    for col in feature_cols:
        if col in train_processed.columns:
            train_processed[col].fillna(train_processed[col].median(), inplace=True)
        if col in test_processed.columns:
            test_processed[col].fillna(test_processed[col].median(), inplace=True)
    
    # Asegurar que test tiene todas las columnas
    missing_in_test = set(feature_cols) - set(test_processed.columns)
    if missing_in_test:
        print(f"   ⚠️  Completando {len(missing_in_test)} columnas faltantes en test...")
        for col in missing_in_test:
            test_processed[col] = train_processed[col].median()
    
    X_train = train_processed[feature_cols]
    y_train = train_processed[config.TARGET]
    X_test = test_processed[feature_cols]
    
    print(f"   ✓ Features: {len(feature_cols)}")
    print(f"   ✓ X_train: {X_train.shape}")
    print(f"   ✓ X_test: {X_test.shape}")
    
    return X_train, y_train, X_test, test_processed['ID'], feature_cols


def train_improved_model(X_train, y_train):
    """Entrena modelo con parámetros anti-overfitting"""
    print("\n🤖 Entrenando modelo mejorado (anti-overfitting)...")
    
    # Parámetros con más regularización
    improved_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,  # Reducido de 8 para evitar overfitting
        'learning_rate': 0.03,  # Más bajo para aprender mejor
        'n_estimators': 2000,  # Más árboles pero más simples
        'subsample': 0.7,  # Más agresivo
        'colsample_bytree': 0.7,  # Más agresivo
        'colsample_bylevel': 0.7,
        'min_child_weight': 5,  # Aumentado
        'gamma': 0.1,  # Añadir penalización
        'reg_alpha': 0.5,  # Más regularización L1
        'reg_lambda': 2.0,  # Más regularización L2
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1,
        'tree_method': 'hist',
        'early_stopping_rounds': 100  # Más paciencia
    }
    
    predictor = DemandPredictor(params=improved_params, random_state=config.RANDOM_STATE)
    
    print("\n   📊 Validación cruzada con 5 folds...")
    cv_results = predictor.cross_validate(X_train, y_train, n_splits=5, verbose=True)
    
    print("\n   🎯 Entrenamiento final...")
    predictor.train(X_train, y_train, verbose=True)
    
    predictor.save(config.MODELS_DIR)
    
    if predictor.feature_importance is not None:
        print("\n   📈 Top 15 Features:")
        print(predictor.feature_importance.head(15).to_string(index=False))
    
    return predictor, cv_results


def apply_post_processing(predictions, strategy='conservative'):
    """Aplica post-procesamiento a las predicciones"""
    print(f"\n🔧 Aplicando post-procesamiento ({strategy})...")
    
    if strategy == 'conservative':
        # Reducir predicciones un 5% (para evitar sobre-estimación)
        predictions = predictions * 0.95
    elif strategy == 'clip_outliers':
        # Clipear outliers
        q1, q3 = np.percentile(predictions, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        predictions = np.clip(predictions, lower, upper)
    elif strategy == 'smooth':
        # Suavizado hacia la media
        mean_pred = predictions.mean()
        predictions = 0.9 * predictions + 0.1 * mean_pred
    
    # Asegurar valores positivos
    predictions = np.maximum(predictions, 1000)
    
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-process', default='conservative', 
                       choices=['none', 'conservative', 'clip_outliers', 'smooth'])
    args = parser.parse_args()
    
    print("="*60)
    print("🥭 MANGO - Modelo Mejorado (Anti-Overfitting)")
    print("="*60)
    
    # Crear directorios
    for directory in [config.MODELS_DIR, config.SUBMISSIONS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Cargar datos
    train_df, test_df, sample_submission = load_data()
    
    # Preparar features
    X_train, y_train, X_test, test_ids, feature_cols = prepare_features_improved(
        train_df, test_df
    )
    
    # Entrenar modelo
    predictor, cv_results = train_improved_model(X_train, y_train)
    
    # Generar predicciones
    print("\n🔮 Generando predicciones...")
    predictions = predictor.predict(X_test)
    
    # Post-procesamiento
    if args.post_process != 'none':
        predictions = apply_post_processing(predictions, args.post_process)
    
    # Crear submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'Production': predictions.round(0).astype(int)
    })
    
    # Estadísticas
    print(f"\n   📊 Estadísticas:")
    print(f"      - Media: {predictions.mean():.2f}")
    print(f"      - Mediana: {np.median(predictions):.2f}")
    print(f"      - Min: {predictions.min():.2f}")
    print(f"      - Max: {predictions.max():.2f}")
    
    # Guardar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = config.SUBMISSIONS_DIR / f"submission_improved_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"\n   ✅ Submission guardado: {submission_path}")
    
    print("\n" + "="*60)
    print("✨ Proceso completado!")
    print("="*60)
    print(f"\n📋 Resumen:")
    print(f"   - Score CV: {cv_results['custom_score']['mean']:.2f} ± {cv_results['custom_score']['std']:.2f}")
    print(f"   - VAR CV: {cv_results['var']['mean']:.4f}")
    print(f"   - RMSE CV: {cv_results['rmse']['mean']:.2f}")
    print("\n🎯 Sube el nuevo archivo a Kaggle!")
    print("="*60)


if __name__ == "__main__":
    main()

