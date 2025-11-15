"""
Script avanzado para evaluar la calidad de las submissions sin labels
Analiza patrones, consistencia y validación contra datos históricos
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

def load_data():
    """Carga datos de train, test y submissions"""
    print("📂 Cargando datos...")
    
    # Cargar train y test
    train_df = pd.read_csv('data/train.csv', sep=';', low_memory=False)
    test_df = pd.read_csv('data/test.csv', sep=';', low_memory=False)
    
    # Cargar submissions
    submissions = {}
    submissions_path = Path('submissions')
    
    for file in submissions_path.glob('*.csv'):
        if file.name == '.gitkeep':
            continue
        try:
            df = pd.read_csv(file)
            submissions[file.stem] = df
        except:
            pass
    
    print(f"   ✓ Train: {train_df.shape}")
    print(f"   ✓ Test: {test_df.shape}")
    print(f"   ✓ Submissions: {len(submissions)}")
    
    return train_df, test_df, submissions

def analyze_historical_patterns(train_df):
    """Analiza patrones históricos de demanda"""
    
    # Calcular estadísticas por familia de producto
    family_stats = train_df.groupby('family').agg({
        'weekly_demand': ['mean', 'std', 'min', 'max', 'median'],
        'Production': ['mean', 'std', 'min', 'max', 'median']
    }).reset_index()
    
    # Calcular estadísticas por precio
    price_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    train_df['price_range'] = pd.cut(train_df['price'], bins=price_bins)
    
    price_stats = train_df.groupby('price_range').agg({
        'weekly_demand': ['mean', 'std', 'min', 'max'],
        'Production': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    return family_stats, price_stats

def validate_against_historical(test_df, submissions, train_df):
    """Valida las predicciones contra patrones históricos"""
    
    results = {}
    
    for name, sub_df in submissions.items():
        # Merge con test para obtener características
        col_name = 'demand' if 'demand' in sub_df.columns else 'Production'
        merged = test_df.merge(sub_df[['ID', col_name]], left_on='ID', right_on='ID', how='left')
        merged.rename(columns={col_name: 'prediction'}, inplace=True)
        
        # Calcular métricas de validación
        validation = {}
        
        # 1. Comparar medias por familia
        pred_by_family = merged.groupby('family')['prediction'].mean()
        hist_by_family = train_df.groupby('family')['Production'].mean()
        
        # Solo familias comunes
        common_families = set(pred_by_family.index) & set(hist_by_family.index)
        
        if len(common_families) > 0:
            family_diff = []
            for fam in common_families:
                if hist_by_family[fam] > 0:
                    diff = abs((pred_by_family[fam] - hist_by_family[fam]) / hist_by_family[fam])
                    family_diff.append(diff)
            
            validation['family_consistency'] = 1 - np.mean(family_diff) if family_diff else 0
        else:
            validation['family_consistency'] = 0
        
        # 2. Rangos razonables (¿están las predicciones en rangos similares al histórico?)
        hist_range = (train_df['Production'].quantile(0.05), train_df['Production'].quantile(0.95))
        pred_in_range = merged['prediction'].between(hist_range[0], hist_range[1]).mean()
        validation['range_consistency'] = pred_in_range
        
        # 3. Coeficiente de variación similar
        hist_cv = train_df['Production'].std() / train_df['Production'].mean()
        pred_cv = merged['prediction'].std() / merged['prediction'].mean()
        validation['cv_similarity'] = 1 - min(abs(hist_cv - pred_cv) / hist_cv, 1)
        
        # 4. Outliers (menos outliers es mejor)
        Q1 = merged['prediction'].quantile(0.25)
        Q3 = merged['prediction'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((merged['prediction'] < (Q1 - 3*IQR)) | (merged['prediction'] > (Q3 + 3*IQR))).sum()
        validation['outlier_ratio'] = 1 - (outliers / len(merged))
        
        # 5. Distribución de predicciones por precio
        if 'price' in merged.columns:
            # Las predicciones deberían correlacionar con precio
            price_pred_corr = merged[['price', 'prediction']].corr().iloc[0, 1]
            validation['price_correlation'] = abs(price_pred_corr) if not np.isnan(price_pred_corr) else 0
        else:
            validation['price_correlation'] = 0
        
        # 6. Varianza: ni muy baja (underfitting) ni muy alta (overfitting)
        hist_var = train_df['Production'].var()
        pred_var = merged['prediction'].var()
        var_ratio = min(pred_var, hist_var) / max(pred_var, hist_var)
        validation['variance_balance'] = var_ratio
        
        # SCORE GLOBAL (promedio ponderado)
        weights = {
            'family_consistency': 0.25,
            'range_consistency': 0.20,
            'cv_similarity': 0.15,
            'outlier_ratio': 0.20,
            'price_correlation': 0.10,
            'variance_balance': 0.10
        }
        
        validation['overall_score'] = sum(validation[k] * weights[k] for k in weights.keys())
        
        results[name] = validation
    
    return results

def create_quality_plots(train_df, test_df, submissions, validation_results):
    """Crea gráficos de análisis de calidad"""
    
    fig = plt.figure(figsize=(24, 16))
    
    # 1. SCORES DE VALIDACIÓN
    ax1 = plt.subplot(3, 4, 1)
    
    scores_df = pd.DataFrame(validation_results).T
    scores_df = scores_df.sort_values('overall_score', ascending=False)
    
    colors = plt.cm.RdYlGn(scores_df['overall_score'])
    bars = ax1.barh(range(len(scores_df)), scores_df['overall_score'], color=colors)
    ax1.set_yticks(range(len(scores_df)))
    ax1.set_yticklabels([name[:20] for name in scores_df.index], fontsize=9)
    ax1.set_xlabel('Score de Calidad', fontsize=11)
    ax1.set_title('🏆 SCORES DE CALIDAD\n(Sin labels - validación histórica)', fontsize=13, fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Añadir valores
    for i, (idx, row) in enumerate(scores_df.iterrows()):
        ax1.text(row['overall_score'] + 0.01, i, f"{row['overall_score']:.3f}", 
                va='center', fontsize=9)
    
    # 2. RADAR CHART DE MÉTRICAS
    ax2 = plt.subplot(3, 4, 2, projection='polar')
    
    metrics = ['family_consistency', 'range_consistency', 'cv_similarity', 
               'outlier_ratio', 'price_correlation', 'variance_balance']
    metrics_labels = ['Consistencia\nFamilia', 'Rango\nRazonable', 'CV\nSimilar',
                      'Sin\nOutliers', 'Correlación\nPrecio', 'Varianza\nBalanceada']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    # Graficar top 3 submissions
    top_3 = scores_df.head(3)
    colors_radar = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (name, row) in enumerate(top_3.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label=name[:15], color=colors_radar[i])
        ax2.fill(angles, values, alpha=0.15, color=colors_radar[i])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics_labels, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_title('📊 Análisis Multi-dimensional\n(Top 3 Submissions)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    ax2.grid(True)
    
    # 3. COMPARACIÓN DE DISTRIBUCIONES CON HISTÓRICO
    ax3 = plt.subplot(3, 4, 3)
    
    # Histograma de train (histórico)
    ax3.hist(train_df['Production'].dropna(), bins=50, alpha=0.3, 
             label='Histórico (Train)', color='gray', density=True)
    
    # Top 2 submissions
    for i, (name, sub_df) in enumerate(list(submissions.items())[:2]):
        col_name = 'demand' if 'demand' in sub_df.columns else 'Production'
        ax3.hist(sub_df[col_name], bins=50, alpha=0.5, 
                label=name[:15], density=True)
    
    ax3.set_xlabel('Valor de Producción', fontsize=11)
    ax3.set_ylabel('Densidad', fontsize=11)
    ax3.set_title('📈 Distribución vs Histórico', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. MATRIZ DE MÉTRICAS DE CALIDAD
    ax4 = plt.subplot(3, 4, 4)
    
    # Crear matriz de scores
    metrics_matrix = scores_df[metrics].T
    sns.heatmap(metrics_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
               ax=ax4, vmin=0, vmax=1, cbar_kws={'label': 'Score'})
    ax4.set_xticklabels([name[:12] for name in metrics_matrix.columns], rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels(metrics_labels, rotation=0, fontsize=9)
    ax4.set_title('🎯 Matriz de Calidad', fontsize=13, fontweight='bold')
    
    # 5. ANÁLISIS POR FAMILIA DE PRODUCTO
    ax5 = plt.subplot(3, 4, 5)
    
    # Comparar predicciones por familia
    top_families = train_df['family'].value_counts().head(10).index
    
    hist_means = train_df[train_df['family'].isin(top_families)].groupby('family')['Production'].mean()
    
    # Merge test con la mejor submission
    best_sub_name = scores_df.index[0]
    best_sub = submissions[best_sub_name]
    col_name = 'demand' if 'demand' in best_sub.columns else 'Production'
    merged = test_df.merge(best_sub[['ID', col_name]], left_on='ID', right_on='ID')
    pred_means = merged[merged['family'].isin(top_families)].groupby('family')[col_name].mean()
    
    x = np.arange(len(top_families))
    width = 0.35
    
    ax5.bar(x - width/2, [hist_means.get(f, 0) for f in top_families], 
           width, label='Histórico', alpha=0.7, color='gray')
    ax5.bar(x + width/2, [pred_means.get(f, 0) for f in top_families], 
           width, label=f'Mejor: {best_sub_name[:12]}', alpha=0.7, color='green')
    
    ax5.set_xlabel('Familia de Producto', fontsize=11)
    ax5.set_ylabel('Producción Media', fontsize=11)
    ax5.set_title('👔 Análisis por Familia\n(Top 10 familias)', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f[:8] for f in top_families], rotation=45, ha='right', fontsize=8)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. ANÁLISIS DE OUTLIERS
    ax6 = plt.subplot(3, 4, 6)
    
    outlier_counts = []
    names = []
    
    for name, sub_df in submissions.items():
        col_name = 'demand' if 'demand' in sub_df.columns else 'Production'
        Q1 = sub_df[col_name].quantile(0.25)
        Q3 = sub_df[col_name].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((sub_df[col_name] < (Q1 - 3*IQR)) | (sub_df[col_name] > (Q3 + 3*IQR))).sum()
        outlier_counts.append(outliers)
        names.append(name[:15])
    
    colors = ['green' if c == min(outlier_counts) else 'orange' for c in outlier_counts]
    ax6.barh(names, outlier_counts, color=colors, alpha=0.7)
    ax6.set_xlabel('Número de Outliers', fontsize=11)
    ax6.set_title('⚠️ Detección de Outliers\n(Menos es mejor)', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. COEFICIENTE DE VARIACIÓN
    ax7 = plt.subplot(3, 4, 7)
    
    cv_values = []
    cv_names = []
    
    hist_cv = train_df['Production'].std() / train_df['Production'].mean()
    
    for name, sub_df in submissions.items():
        col_name = 'demand' if 'demand' in sub_df.columns else 'Production'
        cv = sub_df[col_name].std() / sub_df[col_name].mean()
        cv_values.append(cv)
        cv_names.append(name[:15])
    
    ax7.barh(cv_names, cv_values, alpha=0.7)
    ax7.axvline(x=hist_cv, color='red', linestyle='--', linewidth=2, label=f'Histórico: {hist_cv:.2f}')
    ax7.set_xlabel('Coeficiente de Variación', fontsize=11)
    ax7.set_title('📊 Coeficiente de Variación\n(Comparación con histórico)', fontsize=13, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. RANGO DE PREDICCIONES
    ax8 = plt.subplot(3, 4, 8)
    
    data_for_box = []
    labels_for_box = []
    
    # Añadir histórico
    data_for_box.append(train_df['Production'].dropna())
    labels_for_box.append('Histórico')
    
    for name, sub_df in submissions.items():
        col_name = 'demand' if 'demand' in sub_df.columns else 'Production'
        data_for_box.append(sub_df[col_name])
        labels_for_box.append(name[:12])
    
    bp = ax8.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    
    # Colorear: histórico en gris, resto en colores
    colors_box = ['gray'] + [plt.cm.Set3(i/len(submissions)) for i in range(len(submissions))]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax8.set_ylabel('Valor de Producción', fontsize=11)
    ax8.set_title('📦 Rangos de Predicción', fontsize=13, fontweight='bold')
    ax8.tick_params(axis='x', rotation=45)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. ANÁLISIS POR RANGO DE PRECIO
    ax9 = plt.subplot(3, 4, 9)
    
    # Crear bins de precio
    price_bins = [0, 0.25, 0.5, 0.75, 1.0]
    price_labels = ['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto']
    
    # Histórico
    train_df['price_range'] = pd.cut(train_df['price'], bins=price_bins, labels=price_labels)
    hist_by_price = train_df.groupby('price_range')['Production'].mean()
    
    # Mejor submission
    test_df['price_range'] = pd.cut(test_df['price'], bins=price_bins, labels=price_labels)
    best_sub_name = scores_df.index[0]
    best_sub = submissions[best_sub_name]
    col_name = 'demand' if 'demand' in best_sub.columns else 'Production'
    merged = test_df.merge(best_sub[['ID', col_name]], left_on='ID', right_on='ID')
    pred_by_price = merged.groupby('price_range')[col_name].mean()
    
    x = np.arange(len(price_labels))
    width = 0.35
    
    ax9.bar(x - width/2, [hist_by_price.get(p, 0) for p in price_labels], 
           width, label='Histórico', alpha=0.7, color='gray')
    ax9.bar(x + width/2, [pred_by_price.get(p, 0) for p in price_labels], 
           width, label=f'Mejor: {best_sub_name[:12]}', alpha=0.7, color='green')
    
    ax9.set_xlabel('Rango de Precio', fontsize=11)
    ax9.set_ylabel('Producción Media', fontsize=11)
    ax9.set_title('💰 Análisis por Precio', fontsize=13, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels(price_labels, fontsize=10)
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # 10. ESTABILIDAD ENTRE SUBMISSIONS
    ax10 = plt.subplot(3, 4, 10)
    
    # Crear dataframe con todas las predicciones
    all_preds = {}
    for name, sub_df in submissions.items():
        col_name = 'demand' if 'demand' in sub_df.columns else 'Production'
        all_preds[name[:15]] = sub_df.set_index('ID')[col_name]
    
    df_all = pd.DataFrame(all_preds)
    
    # Calcular desviación estándar por fila (entre submissions)
    stability = df_all.std(axis=1)
    
    ax10.hist(stability, bins=50, color='coral', alpha=0.7)
    ax10.axvline(x=stability.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {stability.mean():.0f}')
    ax10.set_xlabel('Desviación Estándar entre Submissions', fontsize=11)
    ax10.set_ylabel('Frecuencia', fontsize=11)
    ax10.set_title('🎲 Estabilidad de Predicciones\n(Menor variación = más estabilidad)', fontsize=13, fontweight='bold')
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)
    
    # 11. TABLA DE RANKING
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    ranking_text = "🏆 RANKING DE CALIDAD\n" + "="*50 + "\n\n"
    
    for i, (name, row) in enumerate(scores_df.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        ranking_text += f"{medal} {name[:25]}\n"
        ranking_text += f"   Score Global: {row['overall_score']:.3f}\n"
        ranking_text += f"   Consistencia Familia: {row['family_consistency']:.2f}\n"
        ranking_text += f"   Rango Razonable: {row['range_consistency']:.2f}\n"
        ranking_text += f"   Sin Outliers: {row['outlier_ratio']:.2f}\n\n"
    
    ax11.text(0.05, 0.95, ranking_text, transform=ax11.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 12. RECOMENDACIONES
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    best_name = scores_df.index[0]
    best_score = scores_df.iloc[0]['overall_score']
    
    recommendations = f"""
🎯 RECOMENDACIONES

✅ MEJOR SUBMISSION:
   {best_name[:35]}
   Score: {best_score:.3f}

📊 ANÁLISIS:
"""
    
    # Añadir recomendaciones específicas
    if best_score > 0.7:
        recommendations += "\n✅ Excelente calidad predictiva"
    elif best_score > 0.5:
        recommendations += "\n⚠️  Calidad moderada, hay margen de mejora"
    else:
        recommendations += "\n❌ Calidad baja, revisar modelo"
    
    worst_metric = scores_df.iloc[0][metrics].idxmin()
    recommendations += f"\n\n🔍 PUNTO DÉBIL:\n   {worst_metric}\n   Valor: {scores_df.iloc[0][worst_metric]:.3f}"
    
    recommendations += "\n\n💡 PRÓXIMOS PASOS:\n"
    if scores_df.iloc[0]['family_consistency'] < 0.5:
        recommendations += "   • Mejorar features por familia\n"
    if scores_df.iloc[0]['outlier_ratio'] < 0.9:
        recommendations += "   • Aplicar clipping a outliers\n"
    if scores_df.iloc[0]['variance_balance'] < 0.7:
        recommendations += "   • Ajustar regularización\n"
    
    ax12.text(0.05, 0.95, recommendations, transform=ax12.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Guardar
    output_file = 'submissions_quality_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Análisis de calidad guardado: {output_file}")
    
    return fig, scores_df

def main():
    print("="*70)
    print("🔬 ANÁLISIS DE CALIDAD DE SUBMISSIONS (Sin labels de Kaggle)")
    print("="*70)
    print()
    
    # Cargar datos
    train_df, test_df, submissions = load_data()
    
    if len(submissions) == 0:
        print("❌ No se encontraron submissions")
        return
    
    print(f"\n✅ {len(submissions)} submissions cargadas\n")
    
    # Validar contra patrones históricos
    print("🔍 Validando contra patrones históricos...")
    validation_results = validate_against_historical(test_df, submissions, train_df)
    
    # Crear visualizaciones
    print("🎨 Generando análisis visual...")
    fig, scores_df = create_quality_plots(train_df, test_df, submissions, validation_results)
    
    # Guardar resultados
    scores_df.to_csv('submissions_quality_scores.csv')
    print(f"✅ Scores guardados: submissions_quality_scores.csv")
    
    print("\n" + "="*70)
    print("🏆 RANKING FINAL")
    print("="*70)
    print()
    print(scores_df[['overall_score', 'family_consistency', 'range_consistency', 
                     'outlier_ratio', 'price_correlation']].to_string())
    
    print("\n" + "="*70)
    print("💡 MEJOR SUBMISSION RECOMENDADA:")
    print(f"   {scores_df.index[0]}")
    print(f"   Score de Calidad: {scores_df.iloc[0]['overall_score']:.3f}")
    print("="*70)
    
    # Mostrar gráfico
    try:
        plt.show()
    except:
        print("\n⚠️  Abre 'submissions_quality_analysis.png' para ver el análisis")

if __name__ == "__main__":
    main()

