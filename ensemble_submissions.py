"""
Crea ensemble de múltiples submissions para mejorar el score
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

submissions_dir = Path("submissions")

def load_submissions(file_patterns):
    """Carga múltiples archivos de submission"""
    submissions = []
    for pattern in file_patterns:
        files = list(submissions_dir.glob(pattern))
        for f in files:
            df = pd.read_csv(f)
            submissions.append((f.name, df))
            print(f"✓ Cargado: {f.name}")
    return submissions

def create_ensemble(submissions, weights=None, method='weighted_average'):
    """Crea ensemble de submissions"""
    
    if not submissions:
        raise ValueError("No hay submissions para hacer ensemble")
    
    # Verificar que todos tienen el mismo ID
    base_ids = submissions[0][1]['ID']
    for name, df in submissions:
        if not df['ID'].equals(base_ids):
            raise ValueError(f"IDs no coinciden en {name}")
    
    # Extraer predicciones
    predictions = [df['Production'].values for _, df in submissions]
    predictions = np.array(predictions)
    
    if weights is None:
        weights = np.ones(len(submissions)) / len(submissions)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()
    
    if method == 'weighted_average':
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
    elif method == 'median':
        ensemble_pred = np.median(predictions, axis=0)
    elif method == 'geometric_mean':
        ensemble_pred = np.exp(np.average(np.log(predictions + 1), axis=0, weights=weights)) - 1
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    # Crear submission
    ensemble = pd.DataFrame({
        'ID': base_ids,
        'Production': ensemble_pred.round(0).astype(int)
    })
    
    return ensemble

def main():
    print("="*60)
    print("🎯 ENSEMBLE DE SUBMISSIONS")
    print("="*60)
    
    # Intentar cargar diferentes submissions
    print("\n📂 Buscando submissions...")
    
    patterns = [
        "submission_improved_*.csv",
        "submission_202*.csv",
        "submission1.csv",
        "submission2.csv"
    ]
    
    submissions = load_submissions(patterns)
    
    if len(submissions) < 2:
        print("\n❌ Se necesitan al menos 2 submissions para hacer ensemble")
        print("   Ejecuta primero:")
        print("   - python train_improved.py --post-process conservative")
        print("   - python train_improved.py --post-process clip_outliers")
        return
    
    print(f"\n✓ {len(submissions)} submissions cargados")
    
    # Mostrar estadísticas
    print("\n📊 Estadísticas de cada submission:")
    for name, df in submissions:
        print(f"\n{name}:")
        print(f"  Media: {df['Production'].mean():.2f}")
        print(f"  Mediana: {df['Production'].median():.2f}")
        print(f"  Min: {df['Production'].min():.0f}")
        print(f"  Max: {df['Production'].max():.0f}")
    
    # Crear ensembles con diferentes métodos
    methods = ['weighted_average', 'median', 'geometric_mean']
    
    for method in methods:
        print(f"\n🔄 Creando ensemble con método: {method}")
        
        try:
            ensemble = create_ensemble(submissions, method=method)
            
            # Guardar
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_{method}_{timestamp}.csv"
            filepath = submissions_dir / filename
            ensemble.to_csv(filepath, index=False)
            
            print(f"   ✅ Guardado: {filename}")
            print(f"      - Media: {ensemble['Production'].mean():.2f}")
            print(f"      - Mediana: {ensemble['Production'].median():.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Ensemble ponderado (más peso al modelo mejorado)
    if len(submissions) >= 2:
        print("\n🎯 Creando ensemble ponderado (más peso a modelos mejorados)...")
        
        # Dar más peso a submissions con "improved" en el nombre
        weights = []
        for name, _ in submissions:
            if 'improved' in name.lower():
                weights.append(2.0)  # Doble peso
            else:
                weights.append(1.0)
        
        ensemble_weighted = create_ensemble(submissions, weights=weights, method='weighted_average')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ensemble_weighted_smart_{timestamp}.csv"
        filepath = submissions_dir / filename
        ensemble_weighted.to_csv(filepath, index=False)
        
        print(f"   ✅ Guardado: {filename}")
        print(f"      - Media: {ensemble_weighted['Production'].mean():.2f}")
        print(f"      - Pesos usados: {weights}")
    
    print("\n" + "="*60)
    print("✨ Ensemble completado!")
    print("="*60)
    print("\n🎯 Prueba subir los diferentes ensembles a Kaggle")
    print("   y quédate con el que tenga mejor score")


if __name__ == "__main__":
    main()

