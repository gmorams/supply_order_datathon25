"""
Configuración del proyecto de predicción de demanda Mango
"""
import os
from pathlib import Path

# Directorios
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Archivos de datos
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

# Configuración del modelo
RANDOM_STATE = 42
N_SPLITS = 5  # Para validación cruzada
TEST_SIZE = 0.2

# Parámetros de XGBoost (valores iniciales, se optimizarán con Optuna)
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'tree_method': 'hist',
    'early_stopping_rounds': 50
}

# Features categóricas
CATEGORICAL_FEATURES = [
    'id_season',
    'aggregated_family',
    'family',
    'category',
    'fabric',
    'color_name',
    'length_type',
    'silhouette_type',
    'waist_type',
    'sleeve_length_type',
    'heel_shape_type',
    'toecap_type',
    'woven_structure',
    'knit_structure',
    'print_type',
    'archetype',
    'moment',
    'ocassion',
    'has_plus_size'
]

# Features numéricas base
NUMERICAL_FEATURES = [
    'num_stores',
    'num_sizes',
    'price',
    'life_cycle_length',
    'year'
]

# Features de fecha
DATE_FEATURES = [
    'phase_in',
    'phase_out'
]

# Features de embedding de imagen
IMAGE_EMBEDDING_PREFIX = 'image_embedding'

# Columnas a excluir del entrenamiento
EXCLUDE_COLS = [
    'ID',
    'weekly_sales',
    'weekly_demand',
    'Production',
    'num_week_iso',
    'color_rgb'  # Usaremos color_name en su lugar
]

# Target variable
TARGET = 'Production'

# Configuración de optimización
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 3600  # 1 hora

