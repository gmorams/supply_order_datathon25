"""
Script de entrenamiento con validación temporal
Separa una parte del train como validación para evaluar con labels reales
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.feature_engineering import FeatureEngineer
from src.model import DemandPredictor
import config

def load_data():
    """Carga los datos"""
    print("📂 Cargando datos...")
    train_df = pd.read_csv(config.TRAIN_FILE, sep=';', low_memory=False)
    print(f"   ✓ Train completo: {train_df.shape}")
    return train_df

def create_temporal_split(df, validation_ratio=0.2, split_by='season'):
    """
    Crea split temporal para validación
    
    Args:
        df: DataFrame completo
        validation_ratio: Proporción para validación
        split_by: 'season' (por temporada) o 'random' (aleatorio estratificado)
    """
    print(f"\n🔪 Creando split temporal (validación: {validation_ratio*100:.0f}%)...")
    
    if split_by == 'season':
        # Separar por temporada (más realista)
        # Usar la última temporada como validación
        seasons = df['id_season'].unique()
        seasons_sorted = sorted(seasons)
        
        print(f"   • Temporadas disponibles: {len(seasons_sorted)}")
        print(f"   • Temporadas: {seasons_sorted[:5]}... {seasons_sorted[-5:]}")
        
        # Calcular cuántas temporadas usar para validación
        n_val_seasons = max(1, int(len(seasons_sorted) * validation_ratio))
        val_seasons = seasons_sorted[-n_val_seasons:]
        
        print(f"   • Usando últimas {n_val_seasons} temporadas para validación")
        print(f"   • Temporadas de validación: {val_seasons}")
        
        train_mask = ~df['id_season'].isin(val_seasons)
        train_split = df[train_mask].copy()
        val_split = df[~train_mask].copy()
        
    else:  # random stratified
        # Split aleatorio estratificado por familia
        from sklearn.model_selection import train_test_split
        
        train_split, val_split = train_test_split(
            df, 
            test_size=validation_ratio,
            stratify=df['family'] if 'family' in df.columns else None,
            random_state=42
        )
    
    print(f"   ✓ Train: {train_split.shape} ({len(train_split)/len(df)*100:.1f}%)")
    print(f"   ✓ Validación: {val_split.shape} ({len(val_split)/len(df)*100:.1f}%)")
    
    return train_split, val_split

def calculate_kaggle_score(y_true, y_pred, verbose=True):
    """
    Calcula el score similar al de Kaggle
    Penaliza más las ventas perdidas que el exceso de stock
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calcular errores
    errors = y_pred - y_true
    
    # Separar en exceso (overprediction) y faltante (underprediction)
    excess = errors[errors > 0].sum()  # Sobrepredijimos
    shortage = -errors[errors < 0].sum()  # Subpredijimos
    
    # Métricas básicas
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / (y_true + 1))) * 100  # +1 para evitar división por 0
    
    # VAR simulado (ventas / producción)
    # Si predecimos menos que la demanda, perdemos ventas
    # Si predecimos más, tenemos stock sobrante
    actual_sales = np.minimum(y_pred, y_true)  # Solo podemos vender lo que producimos, pero no más de lo que hay demanda
    var = actual_sales.sum() / y_pred.sum() if y_pred.sum() > 0 else 0
    
    # Lost sales (ventas perdidas por falta de stock)
    lost_sales = np.maximum(0, y_true - y_pred).sum()
    lost_sales_pct = (lost_sales / y_true.sum()) * 100
    
    # Excess stock (stock sobrante)
    excess_stock = np.maximum(0, y_pred - y_true).sum()
    excess_stock_pct = (excess_stock / y_pred.sum()) * 100 if y_pred.sum() > 0 else 0
    
    # Score customizado (0-100)
    # Penaliza más las ventas perdidas (factor 2x)
    penalty = (lost_sales * 2 + excess_stock) / y_true.sum()
    score = max(0, 100 * (1 - penalty))
    
    results = {
        'score': score,
        'var': var,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'lost_sales': lost_sales,
        'lost_sales_pct': lost_sales_pct,
        'excess_stock': excess_stock,
        'excess_stock_pct': excess_stock_pct,
        'total_demand': y_true.sum(),
        'total_production': y_pred.sum()
    }
    
    if verbose:
        print("\n" + "="*60)
        print("📊 MÉTRICAS DE EVALUACIÓN")
        print("="*60)
        print(f"🏆 Score Global:        {score:.2f} / 100")
        print(f"📈 VAR:                 {var:.4f} ({var*100:.2f}%)")
        print(f"📏 MAE:                 {mae:.2f}")
        print(f"📏 RMSE:                {rmse:.2f}")
        print(f"📏 MAPE:                {mape:.2f}%")
        print(f"\n💔 Ventas Perdidas:     {lost_sales:.0f} ({lost_sales_pct:.2f}%)")
        print(f"📦 Stock Sobrante:      {excess_stock:.0f} ({excess_stock_pct:.2f}%)")
        print(f"\n📊 Demanda Total:       {y_true.sum():.0f}")
        print(f"🏭 Producción Total:    {y_pred.sum():.0f}")
        print(f"💰 Diferencia:          {y_pred.sum() - y_true.sum():.0f}")
        print("="*60)
    
    return results

def train_and_evaluate(train_split, val_split):
    """Entrena el modelo y evalúa en validación"""
    
    print("\n" + "="*60)
    print("🚀 ENTRENAMIENTO CON VALIDACIÓN TEMPORAL")
    print("="*60)
    
    # Feature Engineering
    print("\n🔧 Feature Engineering...")
    fe = FeatureEngineer()
    
    print("   • Train split...")
    train_processed = fe.fit_transform(train_split, config.CATEGORICAL_FEATURES)
    
    print("   • Validation split...")
    val_processed = fe.transform(val_split, config.CATEGORICAL_FEATURES)
    
    # Preparar datos
    feature_cols = [col for col in train_processed.columns 
                   if col not in ['ID', 'id_season', 'weekly_demand', 'Production', 'weekly_sales']]
    
    X_train = train_processed[feature_cols]
    y_train = train_processed['Production']
    
    X_val = val_processed[feature_cols]
    y_val = val_processed['Production']
    
    print(f"\n   ✓ Features: {len(feature_cols)}")
    print(f"   ✓ X_train: {X_train.shape}")
    print(f"   ✓ X_val: {X_val.shape}")
    
    # Entrenar modelo
    print("\n🤖 Entrenando XGBoost...")
    model = DemandPredictor()
    model.train(X_train, y_train)
    
    # Predecir en validación
    print("\n🔮 Prediciendo en validación...")
    y_pred = model.predict(X_val)
    
    # Evaluar
    print("\n📊 Evaluando predicciones...")
    results = calculate_kaggle_score(y_val, y_pred, verbose=True)
    
    # Análisis adicional
    print("\n" + "="*60)
    print("🔍 ANÁLISIS DETALLADO")
    print("="*60)
    
    # Por familia
    val_analysis = val_processed.copy()
    val_analysis['y_true'] = y_val
    val_analysis['y_pred'] = y_pred
    val_analysis['error'] = y_pred - y_val
    val_analysis['error_pct'] = (val_analysis['error'] / val_analysis['y_true']) * 100
    
    print("\n📊 Por Familia de Producto:")
    family_analysis = val_analysis.groupby('family').agg({
        'y_true': 'sum',
        'y_pred': 'sum',
        'error': 'mean',
        'error_pct': 'mean'
    }).round(2)
    print(family_analysis.head(10))
    
    # Guardar resultados
    val_analysis[['ID', 'family', 'y_true', 'y_pred', 'error', 'error_pct']].to_csv(
        'validation_predictions.csv', index=False
    )
    print("\n✅ Predicciones guardadas: validation_predictions.csv")
    
    return model, results, val_analysis

def create_validation_plots(val_analysis, results):
    """Crea visualizaciones del análisis de validación"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Predicciones vs Real
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(val_analysis['y_true'], val_analysis['y_pred'], alpha=0.5, s=20)
    
    # Línea de predicción perfecta
    min_val = min(val_analysis['y_true'].min(), val_analysis['y_pred'].min())
    max_val = max(val_analysis['y_true'].max(), val_analysis['y_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    ax1.set_xlabel('Demanda Real', fontsize=12)
    ax1.set_ylabel('Predicción', fontsize=12)
    ax1.set_title('🎯 Predicciones vs Real', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Añadir R²
    from sklearn.metrics import r2_score
    r2 = r2_score(val_analysis['y_true'], val_analysis['y_pred'])
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Distribución de errores
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(val_analysis['error'], bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    ax2.axvline(x=val_analysis['error'].mean(), color='blue', linestyle='--', 
               linewidth=2, label=f'Media = {val_analysis["error"].mean():.0f}')
    ax2.set_xlabel('Error (Pred - Real)', fontsize=12)
    ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.set_title('📊 Distribución de Errores', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Errores por familia
    ax3 = plt.subplot(2, 3, 3)
    family_errors = val_analysis.groupby('family')['error_pct'].mean().sort_values()
    top_families = pd.concat([family_errors.head(5), family_errors.tail(5)])
    
    colors = ['green' if x < 0 else 'red' for x in top_families.values]
    ax3.barh(range(len(top_families)), top_families.values, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(top_families)))
    ax3.set_yticklabels([f[:15] for f in top_families.index], fontsize=9)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Error Medio (%)', fontsize=12)
    ax3.set_title('👔 Errores por Familia\n(Verde=Subpredice, Rojo=Sobrepredice)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Métricas principales
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    metrics_text = f"""
🏆 RESULTADOS DE VALIDACIÓN

Score Global:        {results['score']:.2f} / 100
VAR:                 {results['var']:.4f}
MAE:                 {results['mae']:.2f}
RMSE:                {results['rmse']:.2f}
MAPE:                {results['mape']:.2f}%

💔 Ventas Perdidas:  {results['lost_sales']:.0f}
   ({results['lost_sales_pct']:.2f}% de la demanda)

📦 Stock Sobrante:   {results['excess_stock']:.0f}
   ({results['excess_stock_pct']:.2f}% de producción)

📊 Demanda Total:    {results['total_demand']:.0f}
🏭 Producción:       {results['total_production']:.0f}
💰 Diferencia:       {results['total_production'] - results['total_demand']:.0f}

{"✅ Buen balance" if abs(results['total_production'] - results['total_demand']) / results['total_demand'] < 0.1 else "⚠️  Revisar balance producción/demanda"}
"""
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 5. Residuos vs Predicción
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(val_analysis['y_pred'], val_analysis['error'], alpha=0.5, s=20)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Predicción', fontsize=12)
    ax5.set_ylabel('Residuo (Pred - Real)', fontsize=12)
    ax5.set_title('🔍 Análisis de Residuos', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Producción vs Demanda por Producto
    ax6 = plt.subplot(2, 3, 6)
    
    # Tomar muestra aleatoria de productos
    sample_products = val_analysis.sample(min(50, len(val_analysis))).sort_values('y_true')
    x = np.arange(len(sample_products))
    width = 0.35
    
    ax6.bar(x - width/2, sample_products['y_true'], width, label='Demanda Real', alpha=0.7)
    ax6.bar(x + width/2, sample_products['y_pred'], width, label='Predicción', alpha=0.7)
    ax6.set_xlabel('Productos (muestra)', fontsize=12)
    ax6.set_ylabel('Cantidad', fontsize=12)
    ax6.set_title(f'📊 Comparación por Producto\n(Muestra de {len(sample_products)})', fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Guardar
    output_file = 'validation_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Análisis visual guardado: {output_file}")
    
    return fig

def evaluate_existing_submissions(val_split):
    """Evalúa submissions existentes contra el conjunto de validación"""
    
    print("\n" + "="*60)
    print("📊 EVALUANDO SUBMISSIONS EXISTENTES EN VALIDACIÓN")
    print("="*60)
    
    submissions_path = Path('submissions')
    submission_results = []
    
    # Procesar validación
    fe = FeatureEngineer()
    val_processed = fe.transform(val_split, config.CATEGORICAL_FEATURES)
    
    # Obtener IDs de validación
    val_ids = set(val_processed['ID'].values)
    
    for file in submissions_path.glob('*.csv'):
        if file.name == '.gitkeep':
            continue
        
        try:
            sub_df = pd.read_csv(file)
            col_name = 'demand' if 'demand' in sub_df.columns else 'Production'
            
            # Filtrar solo IDs que están en validación
            sub_val = sub_df[sub_df['ID'].isin(val_ids)].copy()
            
            if len(sub_val) == 0:
                print(f"\n⚠️  {file.name}: No hay IDs en común con validación (probablemente para test)")
                continue
            
            # Merge con validación
            merged = val_processed[['ID', 'Production']].merge(
                sub_val[['ID', col_name]], 
                on='ID', 
                how='inner'
            )
            
            if len(merged) == 0:
                continue
            
            # Calcular métricas
            results = calculate_kaggle_score(
                merged['Production'], 
                merged[col_name], 
                verbose=False
            )
            results['name'] = file.stem
            results['n_predictions'] = len(merged)
            
            submission_results.append(results)
            
            print(f"\n✅ {file.name}")
            print(f"   Score: {results['score']:.2f} | VAR: {results['var']:.4f} | MAE: {results['mae']:.2f}")
            
        except Exception as e:
            print(f"\n❌ Error con {file.name}: {e}")
    
    if submission_results:
        # Crear DataFrame de resultados
        results_df = pd.DataFrame(submission_results)
        results_df = results_df.sort_values('score', ascending=False)
        
        # Guardar
        results_df.to_csv('submissions_validation_scores.csv', index=False)
        print(f"\n✅ Resultados guardados: submissions_validation_scores.csv")
        
        # Mostrar ranking
        print("\n" + "="*60)
        print("🏆 RANKING EN VALIDACIÓN")
        print("="*60)
        print(results_df[['name', 'score', 'var', 'mae', 'lost_sales_pct', 'excess_stock_pct']].to_string(index=False))
        
        return results_df
    else:
        print("\n⚠️  No se pudieron evaluar submissions (probablemente son para test, no para validación)")
        return None

def main():
    print("="*60)
    print("🔬 ENTRENAMIENTO CON VALIDACIÓN TEMPORAL")
    print("="*60)
    
    # Cargar datos
    train_df = load_data()
    
    # Crear split temporal
    train_split, val_split = create_temporal_split(
        train_df, 
        validation_ratio=0.2,  # 20% para validación
        split_by='season'  # Separar por temporada (más realista)
    )
    
    # Guardar splits para referencia
    train_split.to_csv('data/train_split.csv', index=False, sep=';')
    val_split.to_csv('data/validation_split.csv', index=False, sep=';')
    print(f"\n✅ Splits guardados:")
    print(f"   • data/train_split.csv")
    print(f"   • data/validation_split.csv")
    
    # Entrenar y evaluar
    model, results, val_analysis = train_and_evaluate(train_split, val_split)
    
    # Crear visualizaciones
    print("\n🎨 Generando visualizaciones...")
    create_validation_plots(val_analysis, results)
    
    # Evaluar submissions existentes (si aplica)
    print("\n🔍 Intentando evaluar submissions existentes...")
    evaluate_existing_submissions(val_split)
    
    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60)
    print("\n📁 Archivos generados:")
    print("   • validation_predictions.csv - Predicciones detalladas")
    print("   • validation_analysis.png - Análisis visual")
    print("   • submissions_validation_scores.csv - Scores de submissions")
    print("   • data/train_split.csv - Datos de entrenamiento")
    print("   • data/validation_split.csv - Datos de validación")
    
    print(f"\n🏆 SCORE EN VALIDACIÓN: {results['score']:.2f} / 100")
    print(f"💡 Este score es una buena aproximación de lo que obtendrás en Kaggle")

if __name__ == "__main__":
    main()

