"""
Script de entrenamiento CON validación local
Separa una parte del train para usarla como test y ver qué tan bueno es el modelo
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.feature_engineering import FeatureEngineer, TARGET
from src.model import DemandPredictor
import config


def calculate_score(y_true, y_pred):
    """
    Calcula el score como lo haría Kaggle
    Penaliza más las ventas perdidas que el exceso de stock
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.maximum(y_pred, 0)  # No negativos
    
    # Ventas perdidas (subproducción)
    lost_sales = np.maximum(0, y_true - y_pred).sum()
    
    # Exceso de stock (sobreproducción)
    excess_stock = np.maximum(0, y_pred - y_true).sum()
    
    # Score: penaliza 2x las ventas perdidas
    penalty = (lost_sales * 2 + excess_stock) / y_true.sum()
    score = max(0, 100 * (1 - penalty))
    
    # Métricas adicionales
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    return {
        'score': score,
        'mae': mae,
        'rmse': rmse,
        'lost_sales': lost_sales,
        'lost_sales_pct': (lost_sales / y_true.sum()) * 100,
        'excess_stock': excess_stock,
        'excess_stock_pct': (excess_stock / y_pred.sum()) * 100 if y_pred.sum() > 0 else 0
    }


def load_and_split_data(test_size=0.2):
    """
    Carga datos y separa en train/test
    """
    print("="*70)
    print("📂 CARGANDO DATOS")
    print("="*70)
    
    # Cargar train completo
    train_df = pd.read_csv(config.TRAIN_FILE, sep=';', low_memory=False)
    print(f"   ✓ Train completo: {train_df.shape}")
    
    # Agrupar por ID (cada producto tiene múltiples semanas)
    # Tomamos la producción única por producto
    print("\n🔄 Agrupando por producto...")
    train_grouped = train_df.groupby('ID').agg({
        TARGET: 'first',  # La producción es la misma para todas las semanas
        **{col: 'first' for col in train_df.columns if col not in ['ID', TARGET, 'weekly_sales', 'weekly_demand']}
    }).reset_index()
    
    print(f"   ✓ Productos únicos: {len(train_grouped)}")
    
    # Split estratificado por familia (si existe)
    print(f"\n🔪 Separando {test_size*100:.0f}% para validación...")
    
    if 'family' in train_grouped.columns:
        train_split, test_split = train_test_split(
            train_grouped,
            test_size=test_size,
            random_state=42,
            stratify=train_grouped['family']
        )
    else:
        train_split, test_split = train_test_split(
            train_grouped,
            test_size=test_size,
            random_state=42
        )
    
    print(f"   ✓ Train: {len(train_split)} productos ({len(train_split)/len(train_grouped)*100:.1f}%)")
    print(f"   ✓ Test:  {len(test_split)} productos ({len(test_split)/len(train_grouped)*100:.1f}%)")
    
    return train_split, test_split


def train_model(train_df):
    """
    Entrena el modelo
    """
    print("\n" + "="*70)
    print("🔧 FEATURE ENGINEERING")
    print("="*70)
    
    fe = FeatureEngineer()
    train_processed = fe.fit_transform(train_df, config.CATEGORICAL_FEATURES)
    
    # Preparar features (solo columnas numéricas)
    exclude_cols = ['ID', 'id_season', 'weekly_demand', TARGET, 'weekly_sales']
    
    # Eliminar columnas categóricas originales (solo quedarnos con las _encoded)
    categorical_original = [col for col in config.CATEGORICAL_FEATURES if col in train_processed.columns]
    exclude_cols.extend(categorical_original)
    
    # Seleccionar solo columnas numéricas
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if train_processed[col].dtype in ['int64', 'float64']]
    
    X_train = train_processed[feature_cols]
    y_train = train_processed[TARGET]
    
    print(f"\n   ✓ Features: {len(feature_cols)}")
    print(f"   ✓ Samples: {len(X_train)}")
    
    # Entrenar
    print("\n" + "="*70)
    print("🤖 ENTRENANDO MODELO")
    print("="*70)
    
    model = DemandPredictor(params=config.XGBOOST_PARAMS)
    model.train(X_train, y_train)
    
    return model, fe, feature_cols


def evaluate_model(model, fe, test_df, feature_cols):
    """
    Evalúa el modelo en el test set
    """
    print("\n" + "="*70)
    print("📊 EVALUANDO EN TEST SET (con labels conocidos)")
    print("="*70)
    
    # Procesar test
    test_processed = fe.transform(test_df, config.CATEGORICAL_FEATURES)
    
    # Asegurar que tenemos las mismas features que en train
    X_test = test_processed[feature_cols]
    y_test = test_processed[TARGET]
    
    print(f"   ✓ Samples test: {len(X_test)}")
    
    # Predecir
    print("\n🔮 Generando predicciones...")
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    print("\n📈 Calculando métricas...")
    metrics = calculate_score(y_test, y_pred)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("🏆 RESULTADOS EN TEST SET")
    print("="*70)
    print(f"\n🎯 SCORE:              {metrics['score']:.2f} / 100")
    print(f"\n📏 MAE:                {metrics['mae']:.2f}")
    print(f"📏 RMSE:               {metrics['rmse']:.2f}")
    print(f"\n💔 Ventas Perdidas:    {metrics['lost_sales']:.0f} ({metrics['lost_sales_pct']:.2f}%)")
    print(f"📦 Stock Sobrante:     {metrics['excess_stock']:.0f} ({metrics['excess_stock_pct']:.2f}%)")
    print("\n" + "="*70)
    
    # Guardar resultados detallados
    results_df = pd.DataFrame({
        'ID': test_processed['ID'],
        'y_true': y_test,
        'y_pred': y_pred,
        'error': y_pred - y_test,
        'error_pct': ((y_pred - y_test) / y_test) * 100
    })
    
    results_df.to_csv('validation_results.csv', index=False)
    print(f"💾 Resultados detallados guardados: validation_results.csv\n")
    
    return metrics, results_df


def save_model(model, fe, metrics):
    """
    Guarda el modelo si es bueno
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar modelo
    model_path = f'models/model_validated_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'feature_engineer': fe}, f)
    
    print(f"💾 Modelo guardado: {model_path}")
    
    # Guardar métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_df['timestamp'] = timestamp
    metrics_df.to_csv(f'models/validation_metrics_{timestamp}.csv', index=False)
    
    print(f"💾 Métricas guardadas: models/validation_metrics_{timestamp}.csv")


def main():
    print("\n" + "="*70)
    print("🎯 ENTRENAMIENTO CON VALIDACIÓN LOCAL")
    print("="*70)
    print("\nEsto te permite ver qué tan bueno es tu modelo ANTES de subir a Kaggle")
    print("Separa 20% del train como test para validar con labels conocidos\n")
    
    # 1. Cargar y separar datos
    train_split, test_split = load_and_split_data(test_size=0.2)
    
    # 2. Entrenar modelo
    model, fe, feature_cols = train_model(train_split)
    
    # 3. Evaluar en test
    metrics, results_df = evaluate_model(model, fe, test_split, feature_cols)
    
    # 4. Guardar modelo
    save_model(model, fe, metrics)
    
    # 5. Interpretación
    print("\n" + "="*70)
    print("💡 INTERPRETACIÓN")
    print("="*70)
    
    score = metrics['score']
    
    if score >= 70:
        print("\n✅ ¡EXCELENTE! Tu modelo funciona muy bien.")
        print("   Este score es una buena estimación de lo que verás en Kaggle.")
    elif score >= 50:
        print("\n⚠️  ACEPTABLE. El modelo funciona pero hay margen de mejora.")
        print("   Considera ajustar hiperparámetros o añadir features simples.")
    else:
        print("\n❌ BAJO. El modelo necesita mejoras.")
        print("   Revisa los hiperparámetros en config.py")
    
    print(f"\n🎯 Score esperado en Kaggle: ~{score:.0f}")
    
    print("\n" + "="*70)
    print("📁 ARCHIVOS GENERADOS:")
    print("="*70)
    print("   • validation_results.csv - Predicciones detalladas")
    print("   • models/model_validated_*.pkl - Modelo entrenado")
    print("   • models/validation_metrics_*.csv - Métricas del modelo")
    
    print("\n" + "="*70)
    print("🚀 PRÓXIMO PASO:")
    print("="*70)
    print("\nSi estás satisfecho con el score, genera la submission final:")
    print("   python predict.py")
    print("\nSi quieres mejorar, ajusta config.py y vuelve a entrenar.")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

