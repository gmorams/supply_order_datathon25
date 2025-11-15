"""
Utilidades generales para el proyecto
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path


def load_data(data_dir: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datasets de train, test y sample submission
    
    Args:
        data_dir: Directorio de datos (por defecto usa config.DATA_DIR)
    
    Returns:
        Tuple con (train_df, test_df, sample_submission)
    """
    if data_dir is None:
        from config import DATA_DIR
        data_dir = DATA_DIR
    
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    
    return train_df, test_df, sample_submission


def save_submission(predictions: np.ndarray, 
                   ids: pd.Series,
                   output_path: Path,
                   round_predictions: bool = True) -> pd.DataFrame:
    """
    Crea y guarda archivo de submission
    
    Args:
        predictions: Array de predicciones
        ids: Series con IDs de productos
        output_path: Ruta donde guardar el archivo
        round_predictions: Si redondear a enteros
    
    Returns:
        DataFrame de submission
    """
    submission = pd.DataFrame({
        'ID': ids,
        'Production': predictions
    })
    
    if round_predictions:
        submission['Production'] = submission['Production'].round(0).astype(int)
    
    # Asegurar que no hay valores negativos
    submission['Production'] = submission['Production'].clip(lower=0)
    
    submission.to_csv(output_path, index=False)
    
    return submission


def calculate_statistics(predictions: np.ndarray) -> Dict:
    """
    Calcula estadísticas descriptivas de predicciones
    
    Args:
        predictions: Array de predicciones
    
    Returns:
        Diccionario con estadísticas
    """
    return {
        'count': len(predictions),
        'mean': float(np.mean(predictions)),
        'median': float(np.median(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'q25': float(np.percentile(predictions, 25)),
        'q75': float(np.percentile(predictions, 75)),
        'sum': float(np.sum(predictions))
    }


def print_statistics(predictions: np.ndarray, title: str = "Estadísticas de Predicciones"):
    """
    Imprime estadísticas de forma formateada
    
    Args:
        predictions: Array de predicciones
        title: Título a mostrar
    """
    stats = calculate_statistics(predictions)
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Productos:    {stats['count']:>10,}")
    print(f"Media:        {stats['mean']:>10,.2f}")
    print(f"Mediana:      {stats['median']:>10,.2f}")
    print(f"Desv. Std:    {stats['std']:>10,.2f}")
    print(f"Mínimo:       {stats['min']:>10,.2f}")
    print(f"Q25:          {stats['q25']:>10,.2f}")
    print(f"Q75:          {stats['q75']:>10,.2f}")
    print(f"Máximo:       {stats['max']:>10,.2f}")
    print(f"Total:        {stats['sum']:>10,.2f}")
    print(f"{'='*60}\n")


def save_model_metadata(metadata: Dict, output_path: Path):
    """
    Guarda metadatos del modelo
    
    Args:
        metadata: Diccionario con metadatos
        output_path: Ruta donde guardar
    """
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_model_metadata(input_path: Path) -> Dict:
    """
    Carga metadatos del modelo
    
    Args:
        input_path: Ruta del archivo de metadatos
    
    Returns:
        Diccionario con metadatos
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def validate_submission(submission: pd.DataFrame) -> bool:
    """
    Valida que el formato del submission sea correcto
    
    Args:
        submission: DataFrame de submission
    
    Returns:
        True si es válido, False si no
    """
    # Verificar columnas
    if list(submission.columns) != ['ID', 'Production']:
        print("❌ Error: Las columnas deben ser ['ID', 'Production']")
        return False
    
    # Verificar que no hay valores faltantes
    if submission.isnull().any().any():
        print("❌ Error: Hay valores faltantes en el submission")
        return False
    
    # Verificar que Production no tiene negativos
    if (submission['Production'] < 0).any():
        print("❌ Error: Hay valores negativos en Production")
        return False
    
    # Verificar que ID no tiene duplicados
    if submission['ID'].duplicated().any():
        print("❌ Error: Hay IDs duplicados")
        return False
    
    print("✅ Submission válido")
    return True


def get_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Agrupa features por tipo
    
    Args:
        df: DataFrame con features
    
    Returns:
        Diccionario con grupos de features
    """
    groups = {
        'numerical': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'embedding': [col for col in df.columns if 'embedding' in col.lower()],
    }
    
    return groups


def memory_usage_report(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Imprime reporte de uso de memoria
    
    Args:
        df: DataFrame a analizar
        name: Nombre del DataFrame
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"\n📊 Reporte de Memoria - {name}")
    print(f"{'='*60}")
    print(f"Filas:        {len(df):>10,}")
    print(f"Columnas:     {len(df.columns):>10,}")
    print(f"Memoria:      {memory_mb:>10.2f} MB")
    print(f"Por fila:     {memory_mb / len(df):>10.4f} MB")
    print(f"{'='*60}\n")


def compare_distributions(train_col: pd.Series, test_col: pd.Series, col_name: str):
    """
    Compara distribuciones entre train y test
    
    Args:
        train_col: Columna de train
        test_col: Columna de test
        col_name: Nombre de la columna
    """
    print(f"\n📊 Comparación de Distribuciones - {col_name}")
    print(f"{'='*60}")
    print(f"{'':20} {'Train':>15} {'Test':>15} {'Diferencia':>15}")
    print(f"{'-'*60}")
    
    metrics = ['mean', 'median', 'std', 'min', 'max']
    
    for metric in metrics:
        train_val = getattr(train_col, metric)()
        test_val = getattr(test_col, metric)()
        diff = test_val - train_val if metric != 'std' else abs(test_val - train_val)
        
        print(f"{metric.capitalize():20} {train_val:>15.2f} {test_val:>15.2f} {diff:>15.2f}")
    
    print(f"{'='*60}\n")


def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detecta outliers en una serie
    
    Args:
        data: Serie de datos
        method: Método de detección ('iqr' o 'zscore')
        threshold: Umbral para detección
    
    Returns:
        Serie booleana indicando outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    else:
        raise ValueError(f"Método no reconocido: {method}")