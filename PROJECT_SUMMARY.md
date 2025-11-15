# 📋 Resumen del Proyecto - Mango Supply Order Datathon 2025

## 🎯 Objetivo

Predecir la cantidad óptima de producción para cada producto de la nueva temporada de Mango, maximizando las ventas a precio completo (VAR) mientras se minimiza el exceso de stock y las ventas perdidas.

## 🏗️ Estructura del Proyecto

```
supply_order_datathon25/
│
├── 📁 data/                          # Datos del datathon
│   ├── train.csv                     # Dataset de entrenamiento (colocar aquí)
│   ├── test.csv                      # Dataset de test (colocar aquí)
│   ├── sample_submission.csv         # Ejemplo de submission (colocar aquí)
│   └── README.md                     # Información sobre los datos
│
├── 📁 models/                        # Modelos entrenados
│   ├── xgboost_model.json           # Modelo XGBoost guardado
│   ├── feature_importance.csv       # Importancia de features
│   └── model_metadata.json          # Metadatos del modelo
│
├── 📁 notebooks/                     # Notebooks de análisis
│   └── 01_exploratory_analysis.ipynb # Análisis exploratorio de datos
│
├── 📁 src/                          # Código fuente
│   ├── __init__.py                  # Inicialización del paquete
│   ├── feature_engineering.py       # Ingeniería de features
│   ├── model.py                     # Implementación del modelo XGBoost
│   └── utils.py                     # Funciones de utilidad
│
├── 📁 submissions/                   # Archivos de submission
│   └── submission_YYYYMMDD_HHMMSS.csv
│
├── 📄 config.py                      # Configuración general
├── 📄 config_custom_example.py       # Ejemplo de configuración personalizada
├── 📄 train.py                       # Script principal de entrenamiento
├── 📄 predict.py                     # Script de predicción
├── 📄 setup.sh                       # Script de configuración inicial
├── 📄 requirements.txt               # Dependencias del proyecto
├── 📄 README.md                      # Documentación principal
├── 📄 QUICK_START.md                 # Guía rápida de inicio
└── 📄 .gitignore                     # Archivos a ignorar en Git
```

## 🚀 Flujo de Trabajo

### 1. Configuración Inicial

```bash
bash setup.sh
# o manualmente:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Exploración de Datos (Opcional)

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### 3. Entrenamiento del Modelo

```bash
# Entrenamiento básico (rápido)
python train.py

# Con optimización de hiperparámetros (mejor rendimiento)
python train.py --optimize
```

### 4. Generación de Predicciones

Las predicciones se generan automáticamente durante el entrenamiento.
Para solo generar predicciones con un modelo ya entrenado:

```bash
python predict.py
```

### 5. Submission

El archivo de submission se guarda en `submissions/submission_YYYYMMDD_HHMMSS.csv`
Súbelo a la plataforma del datathon.

## 🔧 Componentes Principales

### Feature Engineering (`src/feature_engineering.py`)

Crea más de 50+ features derivadas:

- **Features Temporales**: mes, trimestre, temporada, etc.
- **Features Agregadas**: estadísticas por familia, categoría, temporada
- **Features de Interacción**: capacidad total, potencial de ingresos, exposición
- **Features de Embeddings**: estadísticas de embeddings de imagen
- **Features de Lag**: producción de temporadas anteriores

### Modelo (`src/model.py`)

Implementación de XGBoost con:

- **Validación Cruzada**: K-Fold con 5 splits
- **Métricas Personalizadas**: Score VAR, ventas perdidas, exceso de stock
- **Optimización**: Optuna para búsqueda de hiperparámetros
- **Early Stopping**: Prevención de overfitting
- **Feature Importance**: Análisis de importancia de features

### Utilidades (`src/utils.py`)

Funciones auxiliares para:

- Carga y guardado de datos
- Cálculo de estadísticas
- Validación de submissions
- Detección de outliers
- Comparación de distribuciones

## 📊 Métricas de Evaluación

| Métrica | Descripción | Objetivo |
|---------|-------------|----------|
| **Custom Score** | Score 0-100 que penaliza ventas perdidas 2x más que exceso | Maximizar |
| **VAR** | Ventas a precio completo / producción | Maximizar |
| **RMSE** | Root Mean Squared Error | Minimizar |
| **MAE** | Mean Absolute Error | Minimizar |
| **R²** | Coeficiente de determinación | Maximizar |
| **Lost Sales** | Ventas perdidas promedio por producto | Minimizar |
| **Excess Stock** | Exceso de stock promedio por producto | Minimizar |

## 🎛️ Configuración de Parámetros

### Parámetros de XGBoost (en `config.py`)

```python
XGBOOST_PARAMS = {
    'max_depth': 8,              # Profundidad del árbol
    'learning_rate': 0.05,       # Tasa de aprendizaje
    'n_estimators': 1000,        # Número de árboles
    'subsample': 0.8,            # Proporción de muestras
    'colsample_bytree': 0.8,     # Proporción de features
    # ... más parámetros
}
```

### Personalización

Para personalizar parámetros:

1. Copia `config_custom_example.py` a `config_custom.py`
2. Modifica los parámetros según tus necesidades
3. Importa desde `config_custom` en lugar de `config`

## 🔬 Mejoras Implementadas

### ✅ Completado

- [x] Feature engineering completo (temporal, agregado, interacción, lag)
- [x] Modelo XGBoost con validación cruzada
- [x] Métricas personalizadas (VAR, lost sales, excess stock)
- [x] Optimización de hiperparámetros con Optuna
- [x] Feature importance analysis
- [x] Early stopping y regularización
- [x] Scripts de entrenamiento y predicción
- [x] Documentación completa
- [x] Notebook de análisis exploratorio

### 🔄 Posibles Mejoras Futuras

- [ ] Ensemble con LightGBM y CatBoost
- [ ] Modelos específicos por familia de producto
- [ ] Feature selection automático
- [ ] Transfer learning con embeddings
- [ ] Calibración de predicciones
- [ ] Validación temporal estratificada
- [ ] Análisis de errores por segmento
- [ ] Dashboard interactivo de resultados

## 📈 Resultados Esperados

Con la configuración por defecto, deberías obtener:

- **Custom Score**: 70-85 (en validación cruzada)
- **VAR**: 0.75-0.90
- **RMSE**: Variable según escala de datos
- **R²**: 0.6-0.8

Con optimización de hiperparámetros:

- **Custom Score**: 80-90
- **VAR**: 0.80-0.95
- **R²**: 0.7-0.85

## 🛠️ Dependencias Principales

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| pandas | 2.1.4 | Manipulación de datos |
| numpy | 1.26.2 | Operaciones numéricas |
| xgboost | 2.0.3 | Modelo principal |
| scikit-learn | 1.3.2 | Utilidades de ML |
| optuna | 3.5.0 | Optimización de hiperparámetros |
| matplotlib | 3.8.2 | Visualización |
| seaborn | 0.13.0 | Visualización estadística |
| plotly | 5.18.0 | Visualización interactiva |

## 💡 Consejos y Mejores Prácticas

### Para Mejor Performance

1. **Usa optimización de hiperparámetros**: `python train.py --optimize`
2. **Analiza feature importance**: Identifica y enfócate en features importantes
3. **Experimenta con features**: Modifica `src/feature_engineering.py`
4. **Valida con datos temporales**: Asegura que el modelo generaliza bien

### Para Debugging

1. **Revisa notebooks**: Analiza datos en `01_exploratory_analysis.ipynb`
2. **Verifica distribuciones**: Compara train vs test
3. **Analiza errores**: Identifica patrones en predicciones incorrectas
4. **Valida submission**: Usa `validate_submission()` de `utils.py`

### Para Experimentación Rápida

1. **Reduce datos**: Usa subset para iteración rápida
2. **Reduce CV splits**: Usa 3 en lugar de 5 splits
3. **Reduce early_stopping_rounds**: Para convergencia más rápida
4. **Deshabilita optimización**: Usa parámetros por defecto

## 🐛 Solución de Problemas Comunes

### Error: "train.csv not found"
**Solución**: Coloca los archivos CSV en la carpeta `data/`

### Error: "ModuleNotFoundError"
**Solución**: `pip install -r requirements.txt`

### Memoria insuficiente
**Solución**: 
- Reduce el dataset: `df = df.sample(frac=0.8)`
- Reduce `n_estimators` en config
- Usa `tree_method='hist'` en XGBoost

### Predicciones muy altas/bajas
**Solución**:
- Revisa feature engineering
- Ajusta parámetros de regularización
- Usa `clip` en post-procesamiento

### Overfitting
**Solución**:
- Aumenta regularización (`reg_alpha`, `reg_lambda`)
- Reduce `max_depth`
- Aumenta `min_child_weight`
- Reduce `learning_rate` y aumenta `n_estimators`

## 📚 Referencias y Recursos

### Documentación

- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [Optuna Docs](https://optuna.readthedocs.io/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- [Pandas Docs](https://pandas.pydata.org/)

### Papers y Artículos

- Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"
- Demand Forecasting in Fashion Retail
- Time Series Forecasting with Machine Learning

### Competencias Similares

- Kaggle: "Demand Forecasting"
- Kaggle: "Retail Sales Prediction"
- DrivenData: "Supply Chain Optimization"

## 👥 Contribución

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-feature`
3. Commit cambios: `git commit -m 'Add nueva feature'`
4. Push: `git push origin feature/nueva-feature`
5. Abre un Pull Request

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE)

## 🙏 Agradecimientos

- **Mango** por organizar este desafío
- **Comunidad open-source** por las herramientas
- **Participantes** del datathon

---

**¡Éxito en el datathon! 🥭🚀**

*Para más información, consulta [README.md](README.md) o [QUICK_START.md](QUICK_START.md)*

