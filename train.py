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


def load_data(mode: str = 'deploying'):
    """Carga los datos según el modo"""
    print(f"📂 Cargando datos (modo: {mode})...")
    
    if mode == 'deploying':
        # Modo normal: train completo y test real
        train_df = pd.read_csv(config.TRAIN_FILE, sep=';', low_memory=False)
        test_df = pd.read_csv(config.TEST_FILE, sep=';', low_memory=False)
        sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_FILE)
        
        print(f"   ✓ Train: {train_df.shape}")
        print(f"   ✓ Test: {test_df.shape}")
        
        return train_df, test_df, sample_submission, None
    
    else:  # mode == 'practising'
        # Cargar train completo
        full_train = pd.read_csv(config.TRAIN_FILE, sep=';', low_memory=False)
        
        print(f"   ℹ️  Train completo: {full_train.shape}")
        
        # Obtener IDs únicos
        unique_ids = full_train['ID'].unique()
        print(f"   ℹ️  IDs únicos totales: {len(unique_ids)}")
        
        # Split de IDs (80% train, 20% test)
        np.random.seed(config.RANDOM_STATE)
        test_ids = np.random.choice(unique_ids, 
                                    size=int(len(unique_ids) * config.TEST_SIZE), 
                                    replace=False)
        train_ids = np.array([id for id in unique_ids if id not in test_ids])
        
        print(f"   ℹ️  IDs para subtrain: {len(train_ids)}")
        print(f"   ℹ️  IDs para subtest: {len(test_ids)}")
        
        # Filtrar train y test
        subtrain_df = full_train[full_train['ID'].isin(train_ids)].copy()
        subtest_full = full_train[full_train['ID'].isin(test_ids)].copy()
        
        # Para subtest: mantener solo 1 fila por ID (la última semana o agregar)
        # Agregamos las ventas semanales para tener el Production total real
        subtest_ground_truth = subtest_full.groupby('ID').agg({
            'aggregated_family': 'first',
            'family': 'first',
            'category': 'first',
            'fabric': 'first',
            'color_name': 'first',
            'color_rgb': 'first',
            'image_embedding': 'first',
            'length_type': 'first',
            'silhouette_type': 'first',
            'waist_type': 'first',
            'neck_lapel_type': 'first',
            'sleeve_length_type': 'first',
            'heel_shape_type': 'first',
            'toecap_type': 'first',
            'woven_structure': 'first',
            'knit_structure': 'first',
            'print_type': 'first',
            'archetype': 'first',
            'moment': 'first',
            'phase_in': 'first',
            'phase_out': 'first',
            'life_cycle_length': 'first',
            'num_stores': 'first',
            'num_sizes': 'first',
            'has_plus_sizes': 'first',
            'price': 'first',
            'id_season': 'first',
            'Production': 'first',  # Valor real de producción
            'weekly_demand': 'sum'  # Demanda total (ground truth)
        }).reset_index()
        
        # Crear subtest sin columnas de target (simular test real)
        subtest_df = subtest_ground_truth.drop(['Production', 'weekly_demand'], axis=1)
        
        print(f"   ✓ Subtrain: {subtrain_df.shape}")
        print(f"   ✓ Subtest: {subtest_df.shape}")
        print(f"   ✓ Ground truth guardado para evaluación")
        
        return subtrain_df, subtest_df, None, subtest_ground_truth


def prepare_features(train_df, test_df, mode: str = 'deploying'):
    """Prepara las features para el modelo"""
    print(f"\n🔧 Ingeniería de features (modo: {mode})...")
    
    fe = FeatureEngineer()
    
    # Aplicar transformaciones
    print("   - Creando features temporales...")
    train_processed = fe.fit_transform(train_df, config.CATEGORICAL_FEATURES)
    
    print("   - Aplicando transformaciones a test...")
    test_processed = fe.transform(test_df, config.CATEGORICAL_FEATURES)
    
    # Agregar datos semanales solo si existen (modo deploying)
    if mode == 'deploying' and 'weekly_sales' in train_processed.columns:
        print("   - Agregando datos semanales por producto...")
        
        weekly_agg = train_processed.groupby('ID').agg({
            'weekly_sales': ['sum', 'mean', 'max', 'std'],
            'weekly_demand': ['sum', 'mean', 'max', 'std']
        }).reset_index()
        
        weekly_agg.columns = ['ID'] + [f'{col[0]}_{col[1]}' for col in weekly_agg.columns[1:]]
        train_processed = train_processed.merge(weekly_agg, on='ID', how='left')
        
        # Para test, usar estadísticas globales
        for col in weekly_agg.columns:
            if col != 'ID' and col not in test_processed.columns:
                test_processed[col] = train_processed[col].median()
    
    # Seleccionar features
    exclude_cols = config.EXCLUDE_COLS + ['phase_in', 'phase_out']
    feature_cols = [col for col in train_processed.columns 
                   if col not in exclude_cols and 
                   train_processed[col].dtype in ['int64', 'float64']]
    
    # Asegurar mismas columnas en test
    missing_in_test = set(feature_cols) - set(test_processed.columns)
    if missing_in_test:
        print(f"   ⚠️  Columnas faltantes en test: {missing_in_test}")
        for col in missing_in_test:
            test_processed[col] = 0
    
    # Rellenar NaN
    train_processed[feature_cols] = train_processed[feature_cols].fillna(0)
    test_processed[feature_cols] = test_processed[feature_cols].fillna(0)
    
    # Preparar datos
    X_train = train_processed[feature_cols]
    y_train = train_processed[config.TARGET]
    X_test = test_processed[feature_cols]
    
    print(f"   ✓ Features: {len(feature_cols)}")
    print(f"   ✓ X_train: {X_train.shape}")
    print(f"   ✓ X_test: {X_test.shape}")
    
    return X_train, y_train, X_test, test_processed['ID'], feature_cols, fe


def train_model(X_train, y_train, optimize_params: bool = False, skip_cv: bool = False):
    """Entrena el modelo"""
    print("\n🤖 Entrenando modelo XGBoost...")
    
    if optimize_params:
        print("   🔍 Optimizando hiperparámetros...")
        best_params = optimize_hyperparameters(
            X_train, y_train,
            n_trials=config.OPTUNA_N_TRIALS,
            timeout=config.OPTUNA_TIMEOUT,
            random_state=config.RANDOM_STATE
        )
        params = {**config.XGBOOST_PARAMS, **best_params}
    else:
        params = config.XGBOOST_PARAMS
    
    predictor = DemandPredictor(params=params, random_state=config.RANDOM_STATE)
    
    if not skip_cv:
        print("\n   📊 Validación cruzada...")
        cv_results = predictor.cross_validate(X_train, y_train, 
                                             n_splits=config.N_SPLITS,
                                             verbose=True)
    else:
        cv_results = None
    
    print("\n   🎯 Entrenamiento final...")
    predictor.train(X_train, y_train, verbose=True)
    
    print("\n   💾 Guardando modelo...")
    config.MODELS_DIR.mkdir(exist_ok=True)
    predictor.save(config.MODELS_DIR)
    
    if predictor.feature_importance is not None:
        print("\n   📈 Top 20 Features:")
        print(predictor.feature_importance.head(20).to_string(index=False))
    
    return predictor, cv_results


def make_predictions(predictor, X_test, test_ids, mode: str):
    """Genera predicciones"""
    print("\n🔮 Generando predicciones...")
    
    predictions = predictor.predict(X_test)
    
    submission = pd.DataFrame({
        'ID': test_ids,
        'Production': predictions
    })
    
    submission['Production'] = submission['Production'].round(0).astype(int)
    
    print(f"\n   📊 Estadísticas:")
    print(f"      - Media: {predictions.mean():.2f}")
    print(f"      - Mediana: {np.median(predictions):.2f}")
    print(f"      - Min: {predictions.min():.2f}")
    print(f"      - Max: {predictions.max():.2f}")
    
    # Guardar submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{mode}_{timestamp}.csv"
    submission_path = config.SUBMISSIONS_DIR / filename
    submission.to_csv(submission_path, index=False)
    
    print(f"\n   ✅ Guardado: {submission_path}")
    
    return submission


def evaluate_practising(submission: pd.DataFrame, ground_truth: pd.DataFrame):
    """Evalúa el modelo en modo practising"""
    print("\n" + "="*60)
    print("📊 EVALUACIÓN - MODO PRACTISING")
    print("="*60)
    
    # Merge predicciones con ground truth
    eval_df = submission.merge(ground_truth[['ID', 'Production', 'weekly_demand']], 
                               on='ID', 
                               suffixes=('_pred', '_real'))
    
    y_pred = eval_df['Production_pred'].values
    y_real = eval_df['Production_real'].values
    demand_real = eval_df['weekly_demand'].values
    
    # Calcular métricas según el enunciado
    # VAR = full-price sales / production
    # Sales = min(production_pred, demand_real)
    sales = np.minimum(y_pred, demand_real)
    var = np.mean(sales / (y_pred + 1e-10))
    
    # Lost sales y excess stock
    diff = y_pred - demand_real
    lost_sales = np.sum(np.maximum(-diff, 0))
    excess_stock = np.sum(np.maximum(diff, 0))
    
    lost_sales_per_product = lost_sales / len(y_pred)
    excess_stock_per_product = excess_stock / len(y_pred)
    
    # Score (0-100) - penaliza 2x las ventas perdidas
    penalty = (lost_sales * 2 + excess_stock) / len(y_pred)
    max_penalty = np.mean(demand_real) * 2
    score = max(0, 100 * (1 - penalty / max_penalty))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_pred - demand_real) ** 2))
    
    print(f"\n🎯 RESULTADOS:")
    print(f"   VAR (Ventas/Producción):        {var:.4f}")
    print(f"   Score (0-100):                  {score:.2f}")
    print(f"   RMSE:                           {rmse:.2f}")
    print(f"   Lost Sales (por producto):      {lost_sales_per_product:.2f}")
    print(f"   Excess Stock (por producto):    {excess_stock_per_product:.2f}")
    print(f"   Total Ventas:                   {sales.sum():.0f}")
    print(f"   Total Producción:               {y_pred.sum():.0f}")
    print(f"   Total Demanda Real:             {demand_real.sum():.0f}")
    
    # Guardar resultados
    results = {
        'var': var,
        'score': score,
        'rmse': rmse,
        'lost_sales_per_product': lost_sales_per_product,
        'excess_stock_per_product': excess_stock_per_product
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Predicción de demanda Mango')
    parser.add_argument('--mode', '-m', type=str, default='deploying',
                       choices=['p', 'd', 'practising', 'deploying'],
                       help='Modo: p/practising (evaluación local) o d/deploying (submission final)')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimizar hiperparámetros')
    parser.add_argument('--no-cv', action='store_true',
                       help='Saltar validación cruzada')
    args = parser.parse_args()
    
    # Normalizar modo
    mode = 'practising' if args.mode in ['p', 'practising'] else 'deploying'
    
    print("="*60)
    print(f"🥭 MANGO - Predicción de Demanda [{mode.upper()}]")
    print("="*60)
    
    # Crear directorios
    for directory in [config.MODELS_DIR, config.SUBMISSIONS_DIR]:
        directory.mkdir(exist_ok=True)
    
    # Cargar datos
    train_df, test_df, sample_submission, ground_truth = load_data(mode)
    
    # Preparar features
    X_train, y_train, X_test, test_ids, feature_cols, fe = prepare_features(
        train_df, test_df, mode
    )
    
    # Entrenar modelo
    predictor, cv_results = train_model(X_train, y_train, 
                                       optimize_params=args.optimize,
                                       skip_cv=args.no_cv)
    
    # Generar predicciones
    submission = make_predictions(predictor, X_test, test_ids, mode)
    
    # Evaluación final
    print("\n" + "="*60)
    
    if mode == 'practising':
        # Evaluar con ground truth
        results = evaluate_practising(submission, ground_truth)
    else:
        # Modo deploying: solo mostrar CV results
        if cv_results:
            print("📋 Resumen CV:")
            print(f"   Custom Score: {cv_results['custom_score']['mean']:.2f} ± {cv_results['custom_score']['std']:.2f}")
            print(f"   VAR: {cv_results['var']['mean']:.4f} ± {cv_results['var']['std']:.4f}")
            print(f"   RMSE: {cv_results['rmse']['mean']:.4f} ± {cv_results['rmse']['std']:.4f}")
    
    print("\n✨ ¡Proceso completado!")
    print("="*60)


if __name__ == "__main__":
    main()