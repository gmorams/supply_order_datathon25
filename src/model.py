import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle


class DemandPredictor:
    """Modelo de predicción de demanda usando LightGBM"""
    
    def __init__(self, params: dict = None):
        if params is None:
            # Parámetros optimizados para regresión de demanda
            self.params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 500,
                'random_state': 42
            }
        else:
            self.params = params
        
        self.model = None
        self.feature_names = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Entrena el modelo LightGBM"""
        
        self.feature_names = X_train.columns.tolist()
        
        if X_val is not None and y_val is not None:
            # Entrenar con validación
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            # Entrenar sin validación
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X):
        """Predice la demanda"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        predictions = self.model.predict(X)
        # Asegurar que las predicciones sean no negativas
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def evaluate(self, X, y_true):
        """Evalúa el modelo"""
        y_pred = self.predict(X)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def get_feature_importance(self, top_n: int = 20):
        """Obtiene la importancia de features"""
        if self.model is None:
            return None
        
        importance = self.model.feature_importances_
        feature_importance = sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        return feature_importance[:top_n]
    
    def save_model(self, filepath: str):
        """Guarda el modelo"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'params': self.params
            }, f)
    
    def load_model(self, filepath: str):
        """Carga el modelo"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.params = data['params']


def custom_score(y_true, y_pred, penalty_weight: float = 1.5):
    """
    Métrica personalizada que penaliza más la subpredicción (perder ventas)
    que la sobrepredicción (exceso de stock)
    
    penalty_weight: factor de penalización para underestimation
    """
    errors = y_true - y_pred
    
    # Penalizar más cuando predecimos menos de lo real (perdemos ventas)
    underestimation = np.sum(np.maximum(errors, 0) * penalty_weight)
    overestimation = np.sum(np.maximum(-errors, 0))
    
    total_error = underestimation + overestimation
    total_demand = np.sum(y_true)
    
    # Score de 0 a 100 (100 = perfecto)
    if total_demand == 0:
        return 0
    
    # Normalizar el error
    normalized_error = total_error / total_demand
    
    # Convertir a score (menos error = mejor score)
    score = max(0, 100 * (1 - normalized_error))
    
    return score


def calculate_metrics(y_true, y_pred):
    """Calcula todas las métricas relevantes"""
    
    # Métricas estándar
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Métricas de negocio
    lost_sales = np.maximum(y_true - y_pred, 0)
    excess_stock = np.maximum(y_pred - y_true, 0)
    
    avg_lost_sales = lost_sales.mean()
    avg_excess_stock = excess_stock.mean()
    
    total_lost_sales = lost_sales.sum()
    total_excess_stock = excess_stock.sum()
    
    # Score personalizado
    score = custom_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'avg_lost_sales_per_product': avg_lost_sales,
        'avg_excess_stock_per_product': avg_excess_stock,
        'total_lost_sales': total_lost_sales,
        'total_excess_stock': total_excess_stock,
        'custom_score': score
    }