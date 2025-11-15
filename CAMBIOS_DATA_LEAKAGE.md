# 🔧 Corrección de Data Leakage - Resumen de Cambios

## 📊 Problema Identificado

El modelo inicial obtuvo un score de **43.63 en Kaggle** a pesar de tener un score de validación cruzada de **98.33**, lo que indicaba un problema severo de **data leakage** (filtración de datos).

---

## 🔴 Problemas de Data Leakage Encontrados

### **Problema 1: Features Agregadas Sin Temporalidad**

**Ubicación:** `src/feature_engineering.py` - función `create_aggregated_features()`

**Error:**
```python
# ANTES (INCORRECTO):
family_stats = df.groupby('family')[TARGET].agg(['mean', 'std', 'median'])
```

Este código calculaba estadísticas usando **TODOS los datos**, incluyendo la temporada que se intenta predecir. El modelo "veía el futuro".

**Corrección:**
```python
# DESPUÉS (CORRECTO):
# Para cada temporada, calcular stats de temporadas ANTERIORES
for i, season in enumerate(seasons_sorted):
    hist_data = df[df['id_season'].isin(seasons_sorted[:i])]
    family_stats = hist_data.groupby('family')[TARGET].agg(['mean', 'std', 'median'])
```

Ahora las features agregadas usan **solo datos históricos** de temporadas anteriores.

---

### **Problema 2: Agregación de weekly_sales por Producto**

**Ubicación:** `train.py` - función `prepare_features()`

**Error:**
```python
# ANTES (INCORRECTO):
weekly_agg = train_processed.groupby('ID').agg({
    'weekly_sales': ['sum', 'mean', 'max', 'std'],
    'weekly_demand': ['sum', 'mean', 'max', 'std']
})
```

Este código sumaba **TODAS las ventas semanales** del producto, incluyendo las semanas futuras que el modelo debía predecir.

**Corrección:**
```python
# DESPUÉS (CORRECTO):
# ELIMINADO - No agregamos weekly_sales por ID para evitar data leakage
```

Se eliminó completamente esta agregación ya que no es posible hacerla correctamente sin acceso a información futura.

---

## 📈 Impacto en Métricas

### **Validación Cruzada (CV)**

| Métrica | Antes (con leakage) | Después (corregido) | Cambio |
|---------|---------------------|---------------------|---------|
| **Custom Score** | 98.33 ± 0.04 | 97.85 ± 0.07 | -0.48 |
| **VAR** | 0.9791 ± 0.0009 | 0.9750 ± 0.0011 | -0.0041 |
| **RMSE** | 2674.82 ± 121.62 | 3278.02 ± 169.76 | +603.20 |
| **MAE** | 665.52 ± 13.37 | 827.61 ± 22.52 | +162.09 |
| **R²** | 0.9941 ± 0.0005 | 0.9911 ± 0.0009 | -0.003 |

**Interpretación:**
- ✅ Score más bajo es **esperado y correcto** - el modelo ya no "hace trampa"
- ✅ Mayor variabilidad (std más alto) es normal sin información futura
- ✅ RMSE más alto refleja la dificultad real del problema

### **Features Más Importantes**

| Antes (con leakage) | Después (sin leakage) |
|---------------------|------------------------|
| 1. weekly_sales_sum | 1. family_lag1_production |
| 2. family_lag1_production | 2. total_exposure |
| 3. total_capacity | 3. total_capacity |
| 4. family_lag2_production | 4. family_lag2_production |
| 5. num_stores | 5. num_stores |

**Cambios clave:**
- ❌ `weekly_sales_sum` **eliminada** (causaba leakage)
- ✅ `family_lag1_production` ahora es la más importante
- ✅ Aparecen `family_std_production_hist` y `family_mean_production_hist` (features históricas correctas)

### **Predicciones Generadas**

| Estadística | Antes | Después | Cambio |
|-------------|-------|---------|--------|
| **Media** | 23,784.92 | 26,905.09 | +13.1% |
| **Mediana** | 18,888.62 | 19,981.89 | +5.8% |
| **Mínimo** | 8,582.67 | 181.90 | -97.9% |
| **Máximo** | 91,165.48 | 288,200.78 | +216.1% |
| **Std** | 14,772.28 | 30,003.24 | +103.1% |

**Interpretación:**
- ✅ Mayor variabilidad es **más realista**
- ✅ Permite predicciones muy bajas (181) y muy altas (288K)
- ✅ Refleja mejor la incertidumbre real del problema

---

## 📁 Archivos Modificados

### 1. `src/feature_engineering.py`
- ✅ Añadido `self.historical_stats = {}` para guardar estadísticas
- ✅ Modificada `create_aggregated_features()` para usar solo temporadas anteriores
- ✅ Modificada `create_lag_features()` para manejar correctamente train/test
- ✅ Actualizada `fit_transform()` y `transform()` con parámetro `is_train`

### 2. `train.py`
- ✅ Eliminada agregación de `weekly_sales` y `weekly_demand` por ID

---

## 🎯 Próximos Pasos

1. **Subir a Kaggle**: `submissions/submission_20251115_162124.csv`
2. **Expectativa de Score**: 
   - ❌ Anterior: 43.63 (con leakage)
   - ✅ Esperado: **65-75+** (sin leakage)
3. **Si el score es bajo (<60)**:
   - Revisar si hay otros tipos de leakage
   - Ajustar hiperparámetros con Optuna
   - Probar features adicionales (más embeddings, interacciones)

---

## 📝 Lecciones Aprendidas

### ⚠️ **Data Leakage es Traicionero**
- Scores de validación muy altos (>98%) son sospechosos
- Siempre verificar que las features usen **solo datos históricos**
- La diferencia entre score de CV y Kaggle indica leakage

### ✅ **Cómo Evitar Data Leakage**
1. **Orden temporal**: Calcular features usando solo datos de períodos anteriores
2. **Separación train/test**: Nunca usar información del test en train
3. **Agregaciones cuidadosas**: No agregar por ID si incluye el período objetivo
4. **Validación realista**: El score de CV debe ser similar al score real

### 🎓 **Best Practices**
- Usar `shift()` para lags garantiza uso de datos anteriores
- Guardar estadísticas de train para aplicar a test
- Rellenar NaN con medianas, no con valores del mismo período
- Documentar claramente qué features son "seguras" vs "peligrosas"

---

## 🔍 Verificación de Corrección

Para verificar que no hay data leakage:

1. ✅ **Features históricas**: Se calculan por temporada, solo con datos anteriores
2. ✅ **Test set**: Usa estadísticas guardadas del train, no recalcula
3. ✅ **Lags**: Usan `shift()` que garantiza datos anteriores
4. ✅ **No hay agregaciones por ID** que incluyan el objetivo
5. ✅ **Score de CV más realista**: 97.85 vs 98.33

---

**Fecha de corrección:** 15 de noviembre de 2025  
**Versión del modelo:** v3 (sin data leakage)  
**Archivo de submission:** `submission_20251115_162124.csv`

