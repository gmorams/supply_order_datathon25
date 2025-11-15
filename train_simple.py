"""
Script de entrenamiento SIMPLE para predicción de demanda Mango
Sistema simplificado con XGBoost
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Importar funciones de feature engineering
from src.feature_engineering_simple import prepare_features_simple, TARGET

# Configuración
RANDOM_STATE = 42
N_SPLITS = 5


def load_data():
    """Carga los datos de entrenamiento y test"""
    print("📂 Cargando datos...")
    
    # Los archivos usan punto y coma (;) como delimitador
    train_df = pd.read_csv('data/train.csv', sep=';', low_memory=False)
    test_df = pd.read_csv('data/test.csv', sep=';', low_memory=False)
    sample_submission = pd.read_csv('data/sample_submission.csv')
    
    print(f"   ✓ Train: {train_df.shape}")
    print(f"   ✓ Test: {test_df.shape}")
    
    return train_df, test_df, sample_submission


def calculate_custom_score(y_true, y_pred):
    """
    Calcula el score personalizado de Mango (0-100)
    Penaliza más las ventas perdidas que el exceso de stock
    """
    # VAR = ventas / producción
    var = np.minimum(y_true / (y_pred + 1e-10), 1.0)
    
    # Penalización por subproducción (ventas perdidas)
    underproduction_penalty = np.maximum(0, y_true - y_pred) / (y_true + 1e-10)
    
    # Penalización por sobreproducción (exceso)
    overproduction_penalty = np.maximum(0, y_pred - y_true) / (y_pred + 1e-10) * 0.5
    
    # Score combinado (0-100)
    score = 100 * (1 - (underproduction_penalty + overproduction_penalty).mean())
    
    return max(0, min(100, score))


def train_and_evaluate():
    """Pipeline completo de entrenamiento y evaluación"""
    
    print("\n" + "="*60)
    print("🥭 MANGO - Predicción de Demanda (SIMPLE)")
    print("="*60)
    
    # 1. Cargar datos
    train_df, test_df, sample_submission = load_data()
    
    # 2. Feature engineering
    train_processed, test_processed, feature_cols = prepare_features_simple(
        train_df, 
        test_df,
        embedding_method='statistical'  # Cambiar a 'pca' o 'keep_all' si quieres
    )
    
    # 3. Preparar datos para XGBoost
    print("\n🤖 Preparando datos para XGBoost...")
    
    # Excluir columnas que no son features
    exclude_cols = ['ID', 'Production', 'weekly_sales', 'weekly_demand', 'num_week_iso', 
                    'phase_in', 'phase_out', 'year']
    
    X = train_processed[[col for col in feature_cols if col not in exclude_cols]]
    y = train_processed[TARGET]
    X_test = test_processed[[col for col in X.columns if col in test_processed.columns]]
    
    # Rellenar NaN
    X = X.fillna(0)
    X_test = X_test.fillna(0)
    
    print(f"   ✓ X shape: {X.shape}")
    print(f"   ✓ y shape: {y.shape}")
    print(f"   ✓ X_test shape: {X_test.shape}")
    
    # 4. Validación cruzada
    print("\n📊 Validación Cruzada (5-Fold)...")
    
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    scores = {
        'custom_score': [],
        'var': [],
        'rmse': [],
        'mae': [],
        'r2': []
    }
    
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{N_SPLITS}")
        print(f"{'='*50}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Entrenar XGBoost
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method='hist'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Predicciones
        y_pred = model.predict(X_val)
        y_pred = np.maximum(y_pred, 0)  # No permitir predicciones negativas
        
        # Métricas
        custom_score = calculate_custom_score(y_val.values, y_pred)
        var_score = np.mean(np.minimum(y_val.values / (y_pred + 1e-10), 1.0))
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        scores['custom_score'].append(custom_score)
        scores['var'].append(var_score)
        scores['rmse'].append(rmse)
        scores['mae'].append(mae)
        scores['r2'].append(r2)
        
        print(f"\nFold {fold} - Custom Score: {custom_score:.2f}")
        print(f"Fold {fold} - VAR: {var_score:.4f}")
        print(f"Fold {fold} - RMSE: {rmse:.4f}")
        
        models.append(model)
    
    # 5. Resultados de CV
    print("\n" + "="*50)
    print("RESULTADOS DE VALIDACIÓN CRUZADA")
    print("="*50)
    
    for metric, values in scores.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric}: {mean_val:.4f} (+/- {std_val:.4f})")
    
    # 6. Entrenamiento final
    print("\n🎯 Entrenamiento final en todos los datos...")
    
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist'
    )
    
    final_model.fit(X, y, verbose=100)
    
    # 7. Feature importance
    print("\n📈 Top 20 Features más importantes:")
    
    importance_dict = final_model.get_booster().get_score(importance_type='gain')
    if importance_dict:
        feature_importance = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False).head(20)
        
        for idx, row in feature_importance.iterrows():
            print(f"   {row['feature']}: {row['importance']:.2e}")
    
    # 8. Predicciones
    print("\n🔮 Generando predicciones...")
    
    predictions = final_model.predict(X_test)
    predictions = np.maximum(predictions, 0)  # No permitir predicciones negativas
    
    # Estadísticas de predicciones
    print("\n   📊 Estadísticas de predicciones:")
    print(f"      - Media: {predictions.mean():.2f}")
    print(f"      - Mediana: {np.median(predictions):.2f}")
    print(f"      - Min: {predictions.min():.2f}")
    print(f"      - Max: {predictions.max():.2f}")
    print(f"      - Std: {predictions.std():.2f}")
    
    # 9. Crear submission
    submission = pd.DataFrame({
        'ID': test_processed['ID'],
        'Production': predictions
    })
    
    # Guardar
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'submissions/submission_simple_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n   ✅ Submission guardado en: {submission_path}")
    
    # 10. Guardar modelo
    import pickle
    model_path = f'models/xgboost_simple_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"   ✅ Modelo guardado en: {model_path}")
    
    print("\n" + "="*60)
    print("✨ ¡Proceso completado exitosamente!")
    print("="*60)
    print(f"\n📋 Resumen de resultados:")
    print(f"   - Custom Score (CV): {np.mean(scores['custom_score']):.2f} ± {np.std(scores['custom_score']):.2f}")
    print(f"   - VAR (CV): {np.mean(scores['var']):.4f} ± {np.std(scores['var']):.4f}")
    print(f"   - RMSE (CV): {np.mean(scores['rmse']):.2f} ± {np.std(scores['rmse']):.2f}")
    print(f"\n🎯 Archivo para subir a Kaggle: {submission_path}")
    print("="*60)


if __name__ == "__main__":
    train_and_evaluate()

