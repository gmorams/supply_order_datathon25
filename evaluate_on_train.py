"""
Script para evaluar el modelo contra una muestra del dataset de train
Esto nos da una idea del score real antes de subir a Kaggle
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from src.feature_engineering import FeatureEngineer
from src.model import DemandPredictor
import config

def calculate_metrics(y_true, y_pred):
    """Calcula métricas detalladas"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Asegurar valores positivos
    y_pred = np.maximum(y_pred, 0)
    
    # Calcular errores
    errors = y_pred - y_true
    
    # MAE, RMSE, MAPE
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / (y_true + 1))) * 100
    
    # VAR (Ventas / Producción)
    actual_sales = np.minimum(y_pred, y_true)
    var = actual_sales.sum() / y_pred.sum() if y_pred.sum() > 0 else 0
    
    # Lost sales y excess stock
    lost_sales = np.maximum(0, y_true - y_pred).sum()
    excess_stock = np.maximum(0, y_pred - y_true).sum()
    
    # Score personalizado (0-100)
    penalty = (lost_sales * 2 + excess_stock) / y_true.sum()
    score = max(0, 100 * (1 - penalty))
    
    # R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    return {
        'score': score,
        'var': var,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'lost_sales': lost_sales,
        'lost_sales_pct': (lost_sales / y_true.sum()) * 100,
        'excess_stock': excess_stock,
        'excess_stock_pct': (excess_stock / y_pred.sum()) * 100 if y_pred.sum() > 0 else 0,
        'total_demand': y_true.sum(),
        'total_production': y_pred.sum()
    }

def evaluate_trained_model():
    """
    Evalúa el modelo ya entrenado contra el dataset de train
    Simula lo que pasaría en Kaggle
    """
    
    print("="*70)
    print("🔬 EVALUACIÓN DEL MODELO CONTRA DATOS DE TRAIN")
    print("="*70)
    print()
    
    # Cargar datos originales
    print("📂 Cargando datos...")
    train_df = pd.read_csv(config.TRAIN_FILE, sep=';', low_memory=False)
    print(f"   ✓ Train completo: {train_df.shape}")
    
    # Tomar últimas temporadas como "validación"
    seasons = sorted(train_df['id_season'].unique())
    val_season = seasons[-1]  # Última temporada
    
    print(f"\n🔍 Usando temporada {val_season} como validación")
    
    train_subset = train_df[train_df['id_season'] != val_season].copy()
    val_subset = train_df[train_df['id_season'] == val_season].copy()
    
    print(f"   • Train: {len(train_subset)} productos ({len(train_subset)/len(train_df)*100:.1f}%)")
    print(f"   • Validación: {len(val_subset)} productos ({len(val_subset)/len(train_df)*100:.1f}%)")
    
    # Agrupar por producto (ID) - cada producto tiene múltiples semanas
    print("\n🔄 Agrupando por producto...")
    val_by_product = val_subset.groupby('ID').agg({
        'Production': 'first',  # La producción es la misma para todas las semanas
        'family': 'first',
        'price': 'first',
        'num_stores': 'first'
    }).reset_index()
    
    print(f"   ✓ {len(val_by_product)} productos únicos en validación")
    
    # Cargar modelo entrenado
    print("\n🤖 Buscando modelo entrenado...")
    
    model_path = Path('models')
    model_files = list(model_path.glob('demand_model_*.pkl'))
    
    if not model_files:
        print("❌ No se encontró modelo entrenado. Ejecuta train.py primero.")
        return None
    
    # Tomar el más reciente
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"   ✓ Modelo encontrado: {latest_model.name}")
    
    # Cargar modelo
    with open(latest_model, 'rb') as f:
        model = pickle.load(f)
    
    print("   ✓ Modelo cargado")
    
    # Cargar features procesadas guardadas
    print("\n📊 Cargando features procesadas...")
    
    # Intentar cargar el train procesado
    train_processed_path = 'data/train_processed.csv'
    if not Path(train_processed_path).exists():
        print("⚠️  No se encontró train_processed.csv")
        print("   Ejecutando feature engineering...")
        
        fe = FeatureEngineer()
        train_full_processed = fe.fit_transform(train_df, config.CATEGORICAL_FEATURES)
        train_full_processed.to_csv(train_processed_path, index=False)
    else:
        train_full_processed = pd.read_csv(train_processed_path)
    
    print(f"   ✓ Features cargadas: {train_full_processed.shape}")
    
    # Filtrar validación
    val_processed = train_full_processed[train_full_processed['id_season'] == val_season].copy()
    
    # Agrupar por producto (tomar primera fila de cada ID)
    val_processed_by_product = val_processed.groupby('ID').first().reset_index()
    
    # Obtener features
    feature_cols = [col for col in val_processed_by_product.columns 
                   if col not in ['ID', 'id_season', 'weekly_demand', 'Production', 'weekly_sales']]
    
    X_val = val_processed_by_product[feature_cols]
    y_val = val_processed_by_product['Production']
    
    print(f"   ✓ X_val: {X_val.shape}")
    print(f"   ✓ y_val: {y_val.shape}")
    
    # Predecir
    print("\n🔮 Generando predicciones...")
    y_pred = model.predict(X_val)
    
    # Calcular métricas
    print("\n📊 Calculando métricas...")
    metrics = calculate_metrics(y_val, y_pred)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("📊 RESULTADOS DE LA EVALUACIÓN")
    print("="*70)
    print(f"\n🏆 SCORE GLOBAL:        {metrics['score']:.2f} / 100")
    print(f"\n📈 VAR:                 {metrics['var']:.4f} ({metrics['var']*100:.2f}%)")
    print(f"📏 R²:                  {metrics['r2']:.4f}")
    print(f"📏 MAE:                 {metrics['mae']:.2f}")
    print(f"📏 RMSE:                {metrics['rmse']:.2f}")
    print(f"📏 MAPE:                {metrics['mape']:.2f}%")
    print(f"\n💔 Ventas Perdidas:     {metrics['lost_sales']:.0f} ({metrics['lost_sales_pct']:.2f}%)")
    print(f"📦 Stock Sobrante:      {metrics['excess_stock']:.0f} ({metrics['excess_stock_pct']:.2f}%)")
    print(f"\n📊 Demanda Total:       {metrics['total_demand']:.0f}")
    print(f"🏭 Producción Total:    {metrics['total_production']:.0f}")
    print(f"💰 Diferencia:          {metrics['total_production'] - metrics['total_demand']:.0f}")
    print("="*70)
    
    # Crear análisis visual
    create_evaluation_plots(val_processed_by_product, y_val, y_pred, metrics)
    
    # Guardar resultados
    results_df = pd.DataFrame({
        'ID': val_processed_by_product['ID'],
        'family': val_processed_by_product['family'] if 'family' in val_processed_by_product.columns else '',
        'y_true': y_val,
        'y_pred': y_pred,
        'error': y_pred - y_val,
        'error_pct': ((y_pred - y_val) / y_val) * 100
    })
    
    results_df.to_csv('train_evaluation_results.csv', index=False)
    print(f"\n✅ Resultados guardados: train_evaluation_results.csv")
    
    # Análisis por familia
    print("\n" + "="*70)
    print("📊 ANÁLISIS POR FAMILIA DE PRODUCTO")
    print("="*70)
    
    family_analysis = results_df.groupby('family').agg({
        'y_true': 'sum',
        'y_pred': 'sum',
        'error': 'mean',
        'error_pct': 'mean'
    }).round(2)
    family_analysis['var'] = (family_analysis['y_pred'].clip(lower=0) / family_analysis['y_pred']).fillna(0)
    
    print(family_analysis.head(15))
    
    return metrics, results_df

def create_evaluation_plots(val_data, y_true, y_pred, metrics):
    """Crea visualizaciones del análisis"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Pred vs Real
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_true, y_pred, alpha=0.5, s=30)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    ax1.set_xlabel('Demanda Real', fontsize=12)
    ax1.set_ylabel('Predicción', fontsize=12)
    ax1.set_title('🎯 Predicciones vs Realidad', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'R² = {metrics["r2"]:.3f}', transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=11)
    
    # 2. Distribución de errores
    ax2 = plt.subplot(2, 3, 2)
    errors = y_pred - y_true
    ax2.hist(errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=errors.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Media: {errors.mean():.0f}')
    ax2.set_xlabel('Error (Pred - Real)', fontsize=12)
    ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.set_title('📊 Distribución de Errores', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Métricas
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    
    metrics_text = f"""
🏆 RESULTADOS

Score Global:  {metrics['score']:.2f} / 100

VAR:           {metrics['var']:.4f}
R²:            {metrics['r2']:.4f}
MAE:           {metrics['mae']:.2f}
RMSE:          {metrics['rmse']:.2f}
MAPE:          {metrics['mape']:.2f}%

Lost Sales:    {metrics['lost_sales_pct']:.2f}%
Excess Stock:  {metrics['excess_stock_pct']:.2f}%

Balance:       
{"✅ Bueno" if abs(metrics['total_production'] - metrics['total_demand']) / metrics['total_demand'] < 0.1 else "⚠️ Revisar"}
"""
    
    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 4. Residuos
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(y_pred, errors, alpha=0.5, s=30)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicción', fontsize=12)
    ax4.set_ylabel('Residuo', fontsize=12)
    ax4.set_title('🔍 Análisis de Residuos', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. QQ Plot
    ax5 = plt.subplot(2, 3, 5)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax5)
    ax5.set_title('📈 Q-Q Plot (Normalidad)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Comparación de distribuciones
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(y_true, bins=50, alpha=0.5, label='Real', color='blue', density=True)
    ax6.hist(y_pred, bins=50, alpha=0.5, label='Predicción', color='orange', density=True)
    ax6.set_xlabel('Valor de Producción', fontsize=12)
    ax6.set_ylabel('Densidad', fontsize=12)
    ax6.set_title('📊 Distribuciones', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = 'train_evaluation_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualización guardada: {output_file}")
    
    return fig

def main():
    metrics, results = evaluate_trained_model()
    
    if metrics:
        print("\n" + "="*70)
        print("💡 INTERPRETACIÓN")
        print("="*70)
        
        if metrics['score'] > 70:
            print("✅ Excelente! Tu modelo tiene buen rendimiento.")
        elif metrics['score'] > 50:
            print("⚠️  El modelo es aceptable, pero hay margen de mejora.")
        else:
            print("❌ El modelo necesita mejoras significativas.")
        
        print(f"\n💡 Este score ({metrics['score']:.2f}) es una estimación de lo que")
        print("   obtendrías en Kaggle con datos similares.")
        
        print("\n📁 Archivos generados:")
        print("   • train_evaluation_results.csv - Predicciones detalladas")
        print("   • train_evaluation_plots.png - Análisis visual")

if __name__ == "__main__":
    main()

