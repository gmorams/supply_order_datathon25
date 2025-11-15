"""
Script para visualizar y comparar múltiples submissions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_submissions(submissions_dir='submissions'):
    """Carga todos los archivos de submission"""
    submissions = {}
    submissions_path = Path(submissions_dir)
    
    for file in submissions_path.glob('*.csv'):
        if file.name == '.gitkeep':
            continue
        try:
            df = pd.read_csv(file)
            submissions[file.stem] = df
            print(f"✓ Cargado: {file.name} - {len(df)} registros")
        except Exception as e:
            print(f"✗ Error al cargar {file.name}: {e}")
    
    return submissions

def create_comparison_plot(submissions):
    """Crea un gráfico de comparación completo"""
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. DISTRIBUCIONES DE PREDICCIONES
    ax1 = plt.subplot(2, 3, 1)
    for name, df in submissions.items():
        if 'demand' in df.columns:
            col_name = 'demand'
        elif 'Production' in df.columns:
            col_name = 'Production'
        else:
            continue
        
        ax1.hist(df[col_name], bins=50, alpha=0.5, label=name)
    
    ax1.set_xlabel('Predicción de Demanda', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title('📊 Distribución de Predicciones por Submission', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. BOXPLOT COMPARATIVO
    ax2 = plt.subplot(2, 3, 2)
    data_for_box = []
    labels_for_box = []
    
    for name, df in submissions.items():
        col_name = 'demand' if 'demand' in df.columns else 'Production'
        data_for_box.append(df[col_name])
        labels_for_box.append(name[:15])  # Acortar nombres
    
    bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    
    # Colorear boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_box)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Predicción de Demanda', fontsize=12)
    ax2.set_title('📦 Distribución Comparativa (Boxplot)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. ESTADÍSTICAS DESCRIPTIVAS
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    
    stats_text = "📈 ESTADÍSTICAS DESCRIPTIVAS\n" + "="*50 + "\n\n"
    
    for name, df in submissions.items():
        col_name = 'demand' if 'demand' in df.columns else 'Production'
        stats = df[col_name].describe()
        
        stats_text += f"{name[:25]}:\n"
        stats_text += f"  Media: {stats['mean']:.2f}\n"
        stats_text += f"  Mediana: {stats['50%']:.2f}\n"
        stats_text += f"  Std: {stats['std']:.2f}\n"
        stats_text += f"  Min: {stats['min']:.2f}\n"
        stats_text += f"  Max: {stats['max']:.2f}\n\n"
    
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 4. COMPARACIÓN DIRECTA (scatter plot)
    if len(submissions) >= 2:
        ax4 = plt.subplot(2, 3, 4)
        
        # Comparar los dos primeros submissions
        names = list(submissions.keys())
        df1 = submissions[names[0]]
        df2 = submissions[names[1]]
        
        col1 = 'demand' if 'demand' in df1.columns else 'Production'
        col2 = 'demand' if 'demand' in df2.columns else 'Production'
        
        # Merge por ID
        merged = pd.merge(df1[['ID', col1]], df2[['ID', col2]], on='ID', suffixes=('_1', '_2'))
        
        ax4.scatter(merged[f'{col1}_1'], merged[f'{col2}_2'], alpha=0.5, s=20)
        ax4.plot([merged[f'{col1}_1'].min(), merged[f'{col1}_1'].max()], 
                 [merged[f'{col1}_1'].min(), merged[f'{col1}_1'].max()], 
                 'r--', linewidth=2, label='Línea de igualdad')
        
        ax4.set_xlabel(f'{names[0][:20]}', fontsize=11)
        ax4.set_ylabel(f'{names[1][:20]}', fontsize=11)
        ax4.set_title('🔄 Comparación Directa', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Calcular correlación
        corr = merged[f'{col1}_1'].corr(merged[f'{col2}_2'])
        ax4.text(0.05, 0.95, f'Correlación: {corr:.3f}', 
                transform=ax4.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. DIFERENCIAS ENTRE SUBMISSIONS
    if len(submissions) >= 2:
        ax5 = plt.subplot(2, 3, 5)
        
        names = list(submissions.keys())
        df1 = submissions[names[0]]
        df2 = submissions[names[1]]
        
        col1 = 'demand' if 'demand' in df1.columns else 'Production'
        col2 = 'demand' if 'demand' in df2.columns else 'Production'
        
        merged = pd.merge(df1[['ID', col1]], df2[['ID', col2]], on='ID', suffixes=('_1', '_2'))
        merged['diff'] = merged[f'{col1}_1'] - merged[f'{col2}_2']
        merged['diff_pct'] = (merged['diff'] / merged[f'{col1}_1']) * 100
        
        ax5.hist(merged['diff_pct'], bins=50, color='coral', alpha=0.7)
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Diferencia (%)', fontsize=12)
        ax5.set_ylabel('Frecuencia', fontsize=12)
        ax5.set_title(f'📉 Diferencias: {names[0][:15]} vs {names[1][:15]}', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Mostrar estadísticas de diferencias
        mean_diff = merged['diff_pct'].mean()
        median_diff = merged['diff_pct'].median()
        ax5.text(0.05, 0.95, f'Media: {mean_diff:.1f}%\nMediana: {median_diff:.1f}%', 
                transform=ax5.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 6. MATRIZ DE CORRELACIÓN
    if len(submissions) >= 2:
        ax6 = plt.subplot(2, 3, 6)
        
        # Crear dataframe con todas las predicciones
        all_preds = {}
        for name, df in submissions.items():
            col_name = 'demand' if 'demand' in df.columns else 'Production'
            all_preds[name[:15]] = df.set_index('ID')[col_name]
        
        df_all = pd.DataFrame(all_preds)
        corr_matrix = df_all.corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   ax=ax6, vmin=0.8, vmax=1.0, center=0.9)
        ax6.set_title('🔗 Matriz de Correlación entre Submissions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_file = 'submissions_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Gráfico guardado como: {output_file}")
    
    return fig

def create_detailed_comparison_table(submissions):
    """Crea una tabla detallada de comparación"""
    
    stats_list = []
    
    for name, df in submissions.items():
        col_name = 'demand' if 'demand' in df.columns else 'Production'
        stats = df[col_name].describe()
        
        stats_dict = {
            'Submission': name,
            'Count': int(stats['count']),
            'Mean': stats['mean'],
            'Median': stats['50%'],
            'Std': stats['std'],
            'Min': stats['min'],
            'Q1': stats['25%'],
            'Q3': stats['75%'],
            'Max': stats['max'],
            'IQR': stats['75%'] - stats['25%'],
            'CV': (stats['std'] / stats['mean']) * 100  # Coeficiente de variación
        }
        stats_list.append(stats_dict)
    
    df_stats = pd.DataFrame(stats_list)
    
    # Guardar como CSV
    df_stats.to_csv('submissions_statistics.csv', index=False)
    print(f"✅ Estadísticas guardadas como: submissions_statistics.csv")
    
    return df_stats

def main():
    print("="*60)
    print("📊 VISUALIZACIÓN DE SUBMISSIONS - MANGO DATATHON")
    print("="*60)
    print()
    
    # Cargar submissions
    submissions = load_submissions()
    
    if len(submissions) == 0:
        print("❌ No se encontraron archivos de submission")
        return
    
    print(f"\n✅ {len(submissions)} submissions cargadas\n")
    
    # Crear visualizaciones
    print("🎨 Generando visualizaciones...")
    fig = create_comparison_plot(submissions)
    
    # Crear tabla de estadísticas
    print("\n📋 Generando tabla de estadísticas...")
    df_stats = create_detailed_comparison_table(submissions)
    
    print("\n" + "="*60)
    print("✅ VISUALIZACIÓN COMPLETADA")
    print("="*60)
    print("\n📁 Archivos generados:")
    print("   • submissions_comparison.png - Gráfico comparativo completo")
    print("   • submissions_statistics.csv - Estadísticas detalladas")
    print("\n💡 Abre 'submissions_comparison.png' para ver el análisis visual")
    
    # Mostrar la tabla en consola
    print("\n" + "="*60)
    print("📊 RESUMEN DE ESTADÍSTICAS")
    print("="*60)
    print(df_stats.to_string(index=False))
    
    # Mostrar el gráfico
    try:
        plt.show()
    except:
        print("\n⚠️  No se pudo mostrar el gráfico interactivamente.")
        print("   Por favor abre el archivo 'submissions_comparison.png'")

if __name__ == "__main__":
    main()

