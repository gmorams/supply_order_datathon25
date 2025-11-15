# 🥭 Mango - Supply Order Datathon 2025

Solución para el desafío de predicción de demanda de Mango usando **XGBoost**.

## 📋 Descripción del Desafío

El objetivo es predecir la cantidad óptima de producción para cada producto de la nueva temporada de Mango. Este es un problema complejo que involucra:

- 📊 Predicción de ventas 9 meses en el futuro
- 🆕 Productos que aún no existen
- ⏱️ Series de tiempo cortas (16 semanas)
- 📈 Identificación de tendencias emergentes

### Métrica de Evaluación

El modelo se evalúa usando **VAR (Ventas Antes de Rebajas)**:

```
VAR = ventas a precio completo / producción
```

La métrica personalizada penaliza más las **ventas perdidas** que el exceso de stock, reflejando el problema real del negocio.

## 🎯 Características del Proyecto

### ✨ Arquitectura del Modelo

- **Algoritmo principal**: XGBoost (Gradient Boosting)
- **Optimización de hiperparámetros**: Optuna
- **Validación**: K-Fold Cross-Validation (5 folds)
- **Feature Engineering**: Más de 50+ features derivadas

### 🔧 Features Implementadas

1. **Features Temporales**:
   - Mes, trimestre, semana del año
   - Tipo de temporada (Primavera-Verano / Otoño-Invierno)
   - Duración del ciclo de vida

2. **Features Agregadas**:
   - Estadísticas por familia de producto
   - Estadísticas por categoría
   - Estadísticas por número de tiendas
   - Estadísticas por temporada

3. **Features de Interacción**:
   - Capacidad total (tiendas × tamaños)
   - Potencial de ingresos (tiendas × precio)
   - Exposición total (semanas × tiendas)

4. **Features de Embeddings**:
   - Estadísticas de embeddings de imagen
   - Similitud entre productos

5. **Features de Lag**:
   - Producción de temporadas anteriores por familia
   - Tendencias temporales

## 🚀 Instalación

### Requisitos Previos

- Python 3.8+
- pip o conda

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/gmorams/supply_order_datathon25.git
cd supply_order_datathon25

# Instalar dependencias
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
supply_order_datathon25/
├── data/                      # Datos (no incluidos en el repo)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/                    # Modelos entrenados
├── notebooks/                 # Notebooks de análisis
│   └── 01_exploratory_analysis.ipynb
├── src/                       # Código fuente
│   ├── feature_engineering.py
│   └── model.py
├── submissions/               # Archivos de submission
├── config.py                  # Configuración del proyecto
├── train.py                   # Script principal de entrenamiento
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## 💻 Uso

### 1. Preparar los Datos

Coloca los archivos del datathon en la carpeta `data/`:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

### 2. Análisis Exploratorio (Opcional)

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### 3. Entrenar el Modelo

#### Entrenamiento Básico

```bash
python train.py
```

#### Entrenamiento con Optimización de Hiperparámetros

```bash
python train.py --optimize
```

Este proceso:
1. ✅ Carga y preprocesa los datos
2. ✅ Crea features adicionales
3. ✅ Entrena el modelo con validación cruzada
4. ✅ Genera predicciones
5. ✅ Crea archivo de submission

### 4. Resultados

Los archivos generados se guardan en:
- **Modelo**: `models/xgboost_model.json`
- **Feature Importance**: `models/feature_importance.csv`
- **Submission**: `submissions/submission_YYYYMMDD_HHMMSS.csv`

## 📊 Resultados Esperados

El modelo está diseñado para:

- ✅ Maximizar el VAR (Ventas Antes de Rebajas)
- ✅ Minimizar ventas perdidas
- ✅ Reducir exceso de stock
- ✅ Adaptarse a diferentes familias de productos
- ✅ Capturar tendencias estacionales

### Métricas de Evaluación

Durante la validación cruzada, el modelo reporta:

| Métrica | Descripción |
|---------|-------------|
| **Custom Score** | Score personalizado (0-100) que penaliza ventas perdidas |
| **VAR** | Ventas a precio completo / producción |
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **R²** | Coeficiente de determinación |
| **Lost Sales** | Ventas perdidas promedio por producto |
| **Excess Stock** | Exceso de stock promedio por producto |

## 🔬 Metodología

### 1. Feature Engineering

```python
from src.feature_engineering import FeatureEngineer

fe = FeatureEngineer()
train_processed = fe.fit_transform(train_df, categorical_features)
test_processed = fe.transform(test_df, categorical_features)
```

### 2. Entrenamiento del Modelo

```python
from src.model import DemandPredictor

predictor = DemandPredictor(params=xgboost_params)
cv_results = predictor.cross_validate(X_train, y_train, n_splits=5)
predictor.train(X_train, y_train)
```

### 3. Generación de Predicciones

```python
predictions = predictor.predict(X_test)
submission = pd.DataFrame({
    'ID': test_ids,
    'Production': predictions
})
```

## 🎛️ Configuración

Los parámetros del modelo se pueden ajustar en `config.py`:

```python
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    # ... más parámetros
}
```

## 🔄 Optimización de Hiperparámetros

El proyecto incluye optimización automática con Optuna:

```python
from src.model import optimize_hyperparameters

best_params = optimize_hyperparameters(
    X_train, y_train,
    n_trials=50,
    timeout=3600
)
```

## 📈 Mejoras Potenciales

### Corto Plazo
- [ ] Ensemble con LightGBM y CatBoost
- [ ] Feature selection automático
- [ ] Calibración de predicciones

### Medio Plazo
- [ ] Modelos específicos por familia de producto
- [ ] Transfer learning con embeddings de imagen
- [ ] Features de similitud entre productos

### Largo Plazo
- [ ] Modelos de series de tiempo (LSTM, Transformer)
- [ ] Incorporar datos externos (tendencias, clima)
- [ ] Sistema de producción con reentrenamiento automático

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Notas Técnicas

### Manejo de Valores Faltantes

- Features numéricas: Imputación con mediana
- Features categóricas: Categoría 'unknown'
- Embeddings: Imputación con 0

### Encoding de Variables Categóricas

- Target encoding para features con alta cardinalidad
- Frequency encoding como alternativa
- Label encoding para XGBoost (maneja categorías nativamente)

### Validación

- K-Fold Cross-Validation estratificado
- Split temporal para validar predicciones futuras
- Validación en subset de test durante el datathon

## 🐛 Solución de Problemas

### Error: "train.csv not found"
Asegúrate de que los archivos de datos están en la carpeta `data/`.

### Error: Memory issues
Reduce el número de features o usa submuestreo:
```python
train_df = train_df.sample(frac=0.8, random_state=42)
```

### Predicciones muy altas/bajas
Ajusta los parámetros del modelo en `config.py` o activa la optimización:
```bash
python train.py --optimize
```

## 📚 Referencias

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 👥 Autores

- **Tu Nombre** - *Desarrollador Principal*

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- Mango por organizar este desafío
- La comunidad de data science por las herramientas open-source
- Todos los participantes del datathon

---

**¡Buena suerte en el datathon! 🚀**

Si tienes preguntas o sugerencias, no dudes en abrir un issue en el repositorio.
