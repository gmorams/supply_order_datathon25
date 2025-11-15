"""
Ejemplo de configuración personalizada
Copia este archivo a config_custom.py y modifica según tus necesidades
"""

# Parámetros personalizados de XGBoost
CUSTOM_XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    
    # Parámetros de estructura del árbol
    'max_depth': 10,              # Profundidad máxima (4-12)
    'min_child_weight': 5,        # Peso mínimo en nodo hijo (1-10)
    'gamma': 0.1,                 # Reducción mínima de pérdida (0-1)
    
    # Parámetros de boosting
    'learning_rate': 0.03,        # Tasa de aprendizaje (0.01-0.3)
    'n_estimators': 1500,         # Número de árboles (500-2000)
    
    # Parámetros de regularización
    'reg_alpha': 0.2,             # Regularización L1 (0-2)
    'reg_lambda': 1.5,            # Regularización L2 (0-2)
    
    # Parámetros de muestreo
    'subsample': 0.85,            # Proporción de muestras por árbol (0.6-1.0)
    'colsample_bytree': 0.85,     # Proporción de features por árbol (0.6-1.0)
    'colsample_bylevel': 0.85,    # Proporción de features por nivel (0.6-1.0)
    
    # Configuración general
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',        # Método de construcción de árboles
    'early_stopping_rounds': 50
}

# Estrategias de validación
VALIDATION_STRATEGIES = {
    'kfold': {
        'n_splits': 5,
        'shuffle': True
    },
    'temporal': {
        'test_seasons': 1,  # Número de temporadas para validación
    }
}

# Features adicionales a crear
CUSTOM_FEATURES = {
    'interaction_features': True,
    'lag_features': True,
    'aggregated_features': True,
    'temporal_features': True,
    'embedding_features': True
}

# Configuración de Optuna
OPTUNA_CONFIG = {
    'n_trials': 100,              # Aumentar para mejor optimización
    'timeout': 7200,              # 2 horas
    'direction': 'maximize',
    'sampler': 'TPE',             # Tree-structured Parzen Estimator
    'pruner': 'MedianPruner'
}

# Post-procesamiento de predicciones
POST_PROCESSING = {
    'clip_min': 0,                # Valor mínimo de predicción
    'clip_max': None,             # Valor máximo (None = sin límite)
    'round_to_integer': True,     # Redondear a enteros
    'apply_smoothing': False,     # Suavizado de predicciones
    'smoothing_factor': 0.1
}

# Ensemble (si quieres combinar múltiples modelos)
ENSEMBLE_CONFIG = {
    'use_ensemble': False,
    'models': ['xgboost', 'lightgbm', 'catboost'],
    'weights': [0.5, 0.3, 0.2]
}

