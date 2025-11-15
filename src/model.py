"""
Modelo XGBoost para predicción de demanda
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from typing import Dict, Tuple, List
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DemandPredictor:
    """Clase principal para entrenamiento y predicción"""
    
    def __init__(self, params: Dict = None, random_state: int = 42):
        self.params = params or {}
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.best_iteration = None
        
    def calculate_custom_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calcula métricas personalizadas para el problema de Mango
        Penaliza más las ventas perdidas que el exceso de stock
        """
        # Evitar valores negativos
        y_pred = np.maximum(y_pred, 0)
        
        # Calcular diferencias
        diff = y_pred - y_true
        
        # Ventas perdidas (cuando predecimos menos de lo real)
        lost_sales = np.sum(np.maximum(-diff, 0))
        lost_sales_per_product = lost_sales / len(y_true)
        
        # Exceso de stock (cuando predecimos más de lo real)
        excess_stock = np.sum(np.maximum(diff, 0))
        excess_stock_per_product = excess_stock / len(y_true)
        
        # Score personalizado (0-100)
        # Penaliza más las ventas perdidas (factor 2x)
        penalty = (lost_sales * 2 + excess_stock) / len(y_true)
        max_penalty = np.mean(y_true) * 2  # Penalización máxima posible
        score = max(0, 100 * (1 - penalty / max_penalty))
        
        # Métricas estándar
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # VAR aproximado (asumiendo que ventas = min(predicción, demanda real))
        simulated_sales = np.minimum(y_pred, y_true)
        var = np.mean(simulated_sales / (y_pred + 1e-10))
        
        return {
            'custom_score': score,
            'var': var,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'lost_sales_per_product': lost_sales_per_product,
            'excess_stock_per_product': excess_stock_per_product
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              verbose: bool = True) -> Dict:
        """Entrena el modelo XGBoost"""
        
        # Configurar parámetros
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': self.random_state,
            'tree_method': 'hist',
            **self.params
        }
        
        # Crear datasets de XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'val'))
        
        # Entrenar modelo
        evals_result = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 1000),
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            verbose_eval=50 if verbose else False
        )
        
        self.best_iteration = self.model.best_iteration
        
        # Calcular feature importance
        importance_dict = self.model.get_score(importance_type='gain')
        if importance_dict:
            # Crear DataFrame con las importancias disponibles
            importance_data = []
            for feature in X_train.columns:
                # Buscar la feature en el diccionario (puede tener formato f0, f1, etc.)
                importance = 0
                for key, value in importance_dict.items():
                    if key == feature or key == f'f{list(X_train.columns).index(feature)}':
                        importance = value
                        break
                importance_data.append({'feature': feature, 'importance': importance})
            self.feature_importance = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
        else:
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': [0] * len(X_train.columns)
            })
        
        # Métricas de entrenamiento
        metrics = {
            'train_rmse': evals_result['train']['rmse'][-1]
        }
        
        if X_val is not None and y_val is not None:
            metrics['val_rmse'] = evals_result['val']['rmse'][-1]
            
            # Predicciones en validación
            y_pred = self.predict(X_val)
            custom_metrics = self.calculate_custom_metric(y_val.values, y_pred)
            metrics.update(custom_metrics)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Genera predicciones"""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest)
        
        # Asegurar que no hay predicciones negativas
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      n_splits: int = 5, verbose: bool = True) -> Dict:
        """Validación cruzada"""
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        cv_scores = {
            'custom_score': [],
            'var': [],
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Fold {fold}/{n_splits}")
                print(f"{'='*50}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Entrenar
            metrics = self.train(X_train_fold, y_train_fold,
                               X_val_fold, y_val_fold,
                               verbose=verbose)
            
            # Guardar métricas
            for key in cv_scores.keys():
                if key in metrics:
                    cv_scores[key].append(metrics[key])
            
            if verbose:
                print(f"\nFold {fold} - Custom Score: {metrics.get('custom_score', 0):.2f}")
                print(f"Fold {fold} - VAR: {metrics.get('var', 0):.4f}")
                print(f"Fold {fold} - RMSE: {metrics.get('rmse', 0):.4f}")
        
        # Calcular promedios
        cv_results = {
            key: {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
            for key, values in cv_scores.items()
        }
        
        if verbose:
            print(f"\n{'='*50}")
            print("RESULTADOS DE VALIDACIÓN CRUZADA")
            print(f"{'='*50}")
            for key, stats in cv_results.items():
                print(f"{key}: {stats['mean']:.4f} (+/- {stats['std']:.4f})")
        
        return cv_results
    
    def save(self, path: Path):
        """Guarda el modelo"""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Guardar modelo XGBoost
        model_path = path / "xgboost_model.json"
        self.model.save_model(str(model_path))
        
        # Guardar feature importance
        if self.feature_importance is not None:
            importance_path = path / "feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
        
        # Guardar metadatos
        metadata = {
            'params': self.params,
            'random_state': self.random_state,
            'best_iteration': self.best_iteration
        }
        metadata_path = path / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, path: Path):
        """Carga el modelo"""
        # Cargar modelo XGBoost
        model_path = path / "xgboost_model.json"
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        
        # Cargar feature importance
        importance_path = path / "feature_importance.csv"
        if importance_path.exists():
            self.feature_importance = pd.read_csv(importance_path)
        
        # Cargar metadatos
        metadata_path = path / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.params = metadata.get('params', {})
                self.random_state = metadata.get('random_state', 42)
                self.best_iteration = metadata.get('best_iteration')


def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series,
                            n_trials: int = 50,
                            timeout: int = 3600,
                            random_state: int = 42) -> Dict:
    """Optimiza hiperparámetros usando Optuna"""
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'early_stopping_rounds': 50
        }
        
        # Crear predictor
        predictor = DemandPredictor(params=params, random_state=random_state)
        
        # Validación cruzada
        cv_results = predictor.cross_validate(X, y, n_splits=3, verbose=False)
        
        # Retornar score personalizado (queremos maximizar)
        return cv_results['custom_score']['mean']
    
    # Crear estudio de Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    print(f"\nMejor score: {study.best_value:.4f}")
    print(f"Mejores parámetros: {study.best_params}")
    
    return study.best_params

