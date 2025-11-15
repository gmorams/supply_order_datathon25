"""
Script para evaluar un modelo ya entrenado
Uso: python evaluate_model.py models/xgboost_simple_20251115_163609.pkl
"""
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

from src.feature_engineering import FeatureEngineer, TARGET
import config

# Importar funciones de train_with_test
from train_with_test import calculate_score, load_and_split_data


def load_saved_model(model_path):
    """Carga un modelo guardado"""
    print(f"📂 Cargando modelo: {model_path}")
    
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    # El modelo puede estar guardado de dos formas:
    if isinstance(saved_data, dict):
        # Formato: {'model': model, 'feature_engineer': fe}
        model = saved_data.get('model')
        fe = saved_data.get('feature_engineer', None)
    else:
        # Formato: solo el modelo
        model = saved_data
        fe = None
    
    print("   ✓ Modelo cargado")
    return model, fe


def evaluate_existing_model(model_path):
    """Evalúa un modelo ya entrenado"""
    
    print("="*70)
    print("🔍 EVALUANDO MODELO EXISTENTE")
    print("="*70)
    
    # 1. Cargar modelo
    model, fe_saved = load_saved_model(model_path)
    
    # 2. Cargar y separar datos
    print("\n📂 Cargando datos...")
    train_split, test_split = load_and_split_data(test_size=0.2)
    
    # 3. Procesar features
    print("\n🔧 Procesando features...")
    
    if fe_saved is not None:
        # Usar el feature engineer guardado con el modelo
        fe = fe_saved
        print("   ✓ Usando feature engineer del modelo guardado")
    else:
        # Crear uno nuevo
        fe = FeatureEngineer()
        train_split_processed = fe.fit_transform(train_split, config.CATEGORICAL_FEATURES)
        print("   ✓ Feature engineer creado nuevo")
    
    # Procesar test
    test_processed = fe.transform(test_split, config.CATEGORICAL_FEATURES)
    
    # Seleccionar features (solo numéricas)
    exclude_cols = ['ID', 'id_season', 'weekly_demand', TARGET, 'weekly_sales']
    categorical_original = [col for col in config.CATEGORICAL_FEATURES if col in test_processed.columns]
    exclude_cols.extend(categorical_original)
    
    feature_cols = [col for col in test_processed.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if test_processed[col].dtype in ['int64', 'float64']]
    
    X_test = test_processed[feature_cols]
    y_test = test_processed[TARGET]
    
    print(f"   ✓ Features: {len(feature_cols)}")
    print(f"   ✓ Test samples: {len(X_test)}")
    
    # 4. Predecir
    print("\n🔮 Generando predicciones...")
    y_pred = model.predict(X_test)
    
    # 5. Calcular métricas
    print("\n📈 Calculando métricas...")
    metrics = calculate_score(y_test, y_pred)
    
    # 6. Mostrar resultados
    print("\n" + "="*70)
    print("🏆 RESULTADOS DEL MODELO")
    print("="*70)
    print(f"\n🎯 SCORE:              {metrics['score']:.2f} / 100")
    print(f"\n📏 MAE:                {metrics['mae']:.2f}")
    print(f"📏 RMSE:               {metrics['rmse']:.2f}")
    print(f"\n💔 Ventas Perdidas:    {metrics['lost_sales']:.0f} ({metrics['lost_sales_pct']:.2f}%)")
    print(f"📦 Stock Sobrante:     {metrics['excess_stock']:.0f} ({metrics['excess_stock_pct']:.2f}%)")
    print("="*70)
    
    # 7. Guardar resultados
    results_df = pd.DataFrame({
        'ID': test_processed['ID'],
        'y_true': y_test,
        'y_pred': y_pred,
        'error': y_pred - y_test,
        'error_pct': ((y_pred - y_test) / y_test) * 100
    })
    
    output_file = f'evaluation_{Path(model_path).stem}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Resultados guardados: {output_file}")
    
    return metrics


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Por defecto, usar el modelo que mencionaste
        model_path = 'models/xgboost_simple_20251115_163609.pkl'
    
    if not Path(model_path).exists():
        print(f"❌ Error: No se encuentra el archivo {model_path}")
        print("\nUso: python evaluate_model.py <ruta_al_modelo.pkl>")
        sys.exit(1)
    
    metrics = evaluate_existing_model(model_path)
    
    print("\n" + "="*70)
    print("✅ EVALUACIÓN COMPLETADA")
    print("="*70)