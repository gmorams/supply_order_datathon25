"""
Configuración del proyecto
"""
from pathlib import Path

# Directorios
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
MODELS_DIR = ROOT_DIR / "models"

# Archivos de datos
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

# Target - CORREGIDO
TARGET = 'demand_total'  # Suma de weekly_demand por producto

# Features a excluir del modelo
EXCLUDE_COLS = ['ID', 'Production', 'weekly_sales', 'weekly_demand', 
                'num_week_iso', 'year', 'image_embedding', 'demand_total']

# Features categóricas para encoding
CATEGORICAL_FEATURES = [
    'aggregated_family', 'family', 'category', 'fabric', 'color_name',
    'length_type', 'silhouette_type', 'waist_type', 'neck_lapel_type',
    'sleeve_length_type', 'heel_shape_type', 'toecap_type', 
    'woven_structure', 'knit_structure', 'print_type', 'archetype', 'moment'
]

# Parámetros del modelo
XGBOOST_PARAMS = {
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'early_stopping_rounds': 50
}

# Validación cruzada
N_SPLITS = 5
RANDOM_STATE = 42

# Optimización de hiperparámetros
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 3600

# Split para modo practising
TEST_SIZE = 0.2