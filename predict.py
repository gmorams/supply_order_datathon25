"""
Script para hacer predicciones con un modelo ya entrenado
"""
import pandas as pd
import sys
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'src'))

from feature_engineering import FeatureEngineer
from model import DemandPredictor
import config


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Genera predicciones con modelo entrenado')
    parser.add_argument('--model-path', type=str, default='models/',
                       help='Ruta al directorio del modelo')
    parser.add_argument('--test-file', type=str, default=None,
                       help='Ruta al archivo de test (por defecto usa config.TEST_FILE)')
    parser.add_argument('--output', type=str, default=None,
                       help='Ruta de salida del archivo de submission')
    args = parser.parse_args()
    
    print("🔮 Generando predicciones...")
    
    # Cargar datos
    test_file = args.test_file if args.test_file else config.TEST_FILE
    test_df = pd.read_csv(test_file)
    print(f"   ✓ Test cargado: {test_df.shape}")
    
    # Cargar train para el feature engineer
    train_df = pd.read_csv(config.TRAIN_FILE)
    print(f"   ✓ Train cargado para feature engineering: {train_df.shape}")
    
    # Preparar features
    print("\n🔧 Preparando features...")
    fe = FeatureEngineer()
    fe.fit_transform(train_df, config.CATEGORICAL_FEATURES)
    test_processed = fe.transform(test_df, config.CATEGORICAL_FEATURES)
    
    # Seleccionar features
    exclude_cols = config.EXCLUDE_COLS + ['phase_in', 'phase_out']
    feature_cols = [col for col in test_processed.columns 
                   if col not in exclude_cols and 
                   test_processed[col].dtype in ['int64', 'float64']]
    
    X_test = test_processed[feature_cols].fillna(0)
    print(f"   ✓ Features preparadas: {X_test.shape}")
    
    # Cargar modelo
    print("\n📦 Cargando modelo...")
    model_path = Path(args.model_path)
    predictor = DemandPredictor()
    predictor.load(model_path)
    print(f"   ✓ Modelo cargado desde: {model_path}")
    
    # Generar predicciones
    print("\n🎯 Generando predicciones...")
    predictions = predictor.predict(X_test)
    
    # Crear submission
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Production': predictions.round(0).astype(int)
    })
    
    # Guardar
    if args.output:
        output_path = Path(args.output)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.SUBMISSIONS_DIR / f"submission_{timestamp}.csv"
    
    submission.to_csv(output_path, index=False)
    print(f"\n✅ Predicciones guardadas en: {output_path}")
    print(f"\n📊 Estadísticas:")
    print(f"   - Productos: {len(predictions)}")
    print(f"   - Media: {predictions.mean():.2f}")
    print(f"   - Mediana: {predictions.median():.2f}")
    print(f"   - Min: {predictions.min():.0f}")
    print(f"   - Max: {predictions.max():.0f}")


if __name__ == "__main__":
    main()

