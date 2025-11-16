import pandas as pd
import numpy as np
import os
from datetime import datetime
from glob import glob

def ensemble_submissions(submission_files, output_dir='outputs'):
    """
    Crea un ensemble (promedio) de múltiples archivos de submission
    """
    
    print("="*80)
    print("ENSEMBLE DE SUBMISSIONS")
    print("="*80)
    
    print(f"\nArchivos a promediar ({len(submission_files)}):")
    for i, file in enumerate(submission_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    # Leer todos los submissions
    dfs = []
    for file in submission_files:
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"     Cargado: {len(df)} predicciones, Total: {df['Production'].sum():,}")
    
    # Verificar que todos tengan los mismos IDs
    ids_base = set(dfs[0]['ID'])
    for i, df in enumerate(dfs[1:], 2):
        if set(df['ID']) != ids_base:
            print(f"\n⚠️  ADVERTENCIA: El archivo {i} tiene IDs diferentes!")
    
    # Crear DataFrame con el promedio
    ensemble_df = dfs[0][['ID']].copy()
    
    # Calcular el promedio de las predicciones
    predictions_array = np.array([df['Production'].values for df in dfs])
    ensemble_df['Production'] = np.mean(predictions_array, axis=0).astype(int)
    
    # Guardar
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'{output_dir}/submission_ensemble_{timestamp}.csv'
    
    ensemble_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("ENSEMBLE GENERADO")
    print("="*80)
    print(f"Archivo: {output_path}")
    print(f"Predicciones: {len(ensemble_df)}")
    print(f"Producción total (promedio): {ensemble_df['Production'].sum():,}")
    
    print(f"\nPrimeras líneas del ensemble:")
    print(ensemble_df.head(10).to_string(index=False))
    
    print(f"\nEstadísticas de las predicciones:")
    print(f"  Min: {ensemble_df['Production'].min():,}")
    print(f"  Max: {ensemble_df['Production'].max():,}")
    print(f"  Media: {ensemble_df['Production'].mean():.2f}")
    print(f"  Mediana: {ensemble_df['Production'].median():.0f}")
    
    return ensemble_df, output_path


if __name__ == "__main__":
    # Obtener los últimos 5 archivos de submission
    submission_files = sorted(glob('outputs/submission_*.csv'), reverse=True)[:5]
    
    if len(submission_files) < 5:
        print(f"Solo se encontraron {len(submission_files)} archivos de submission.")
        print("Se usarán todos los disponibles.")
    
    ensemble_df, output_path = ensemble_submissions(submission_files)

