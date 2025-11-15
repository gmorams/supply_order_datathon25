# Mango Datathon 2025 - Predicción de Demanda

Modelo XGBoost para predecir la cantidad de producción de prendas que Mango debe ordenar para la próxima temporada.

## 📋 Estructura del Proyecto

```
supply_order_datathon25/
├── data/
│   ├── train.csv              # Datos de entrenamiento
│   ├── test.csv               # Datos de test (sin labels)
│   └── sample_submission.csv  # Formato de submission
├── src/
│   ├── feature_engineering.py # Creación de features
│   ├── model.py               # Modelo XGBoost
│   └── utils.py               # Funciones auxiliares
├── models/                    # Modelos entrenados (generado)
├── submissions/               # Submissions generadas (generado)
├── config.py                  # Configuración global
├── train_with_test.py         # Entrenamiento CON validación ⭐
├── train.py                   # Entrenamiento completo
├── predict.py                 # Script de predicción
└── requirements.txt           # Dependencias
```

## 🚀 Inicio Rápido

### 1. Configurar entorno

```bash
# Usar Python 3.11
pyenv shell 3.11.9

# Crear entorno virtual
python3 -m venv venv

# Activar entorno
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Colocar los datos

Coloca los archivos CSV en la carpeta `data/`:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

### 3. Entrenar el modelo

**OPCIÓN A - Con validación (RECOMENDADO para empezar):**

```bash
python train_with_test.py
```

El script:
- Separa 20% del train como test de validación
- Entrena en el 80% restante
- Evalúa en el 20% con labels conocidos
- **Te dice qué score esperar en Kaggle**
- Guarda el modelo en `models/`

⏱️ **Tiempo estimado:** 2-3 minutos

**OPCIÓN B - Entrenamiento completo (para submission final):**

```bash
python train.py
```

El script:
- Usa 100% del train para entrenar
- Entrena con cross-validation
- Guarda el modelo en `models/`

⏱️ **Tiempo estimado:** 3-5 minutos

### 4. Generar predicción

```bash
python predict.py
```

El script:
- Carga el modelo entrenado
- Genera predicciones para el test set
- Guarda la submission en `submissions/submission_YYYYMMDD_HHMMSS.csv`

⏱️ **Tiempo estimado:** 30 segundos

## 📊 ¿Qué hace el modelo?

### Preprocesamiento SIMPLE:

1. **Rellena valores nulos** (mediana para números, 'missing' para texto)
2. **Encoding de categóricas** (convierte texto a números)
3. **¡Nada más!** Sin features complejas

### Modelo:

- **Algoritmo:** XGBoost (Gradient Boosting)
- **Validación:** 5-Fold Cross-Validation
- **Métrica:** Custom score (penaliza más ventas perdidas que exceso de stock)

> 💡 **Filosofía:** Empezar simple. Si funciona, ya habrá tiempo de agregar complejidad.

## 📝 Configuración

Edita `config.py` para ajustar:

```python
# Hiperparámetros del modelo
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    # ...
}

# Features categóricas
CATEGORICAL_FEATURES = [
    'family', 'category', 'fabric', 
    'color_name', 'archetype', # ...
]
```

## 🎯 Resultados

El modelo genera:

1. **Modelo entrenado:** `models/xgboost_model.json`
2. **Feature importance:** `models/feature_importance.csv`
3. **Metadata:** `models/model_metadata.json`
4. **Submission:** `submissions/submission_YYYYMMDD_HHMMSS.csv`

## 📈 Métricas de Evaluación

Durante el entrenamiento verás:

- **Score CV:** Score promedio en cross-validation (0-100)
- **VAR:** Ratio de ventas / producción
- **MAE, RMSE, R²:** Métricas estándar de regresión

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError"
```bash
# Asegúrate de estar en el entorno virtual
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "FileNotFoundError"
```bash
# Verifica que los archivos CSV estén en data/
ls data/
```

### Error: "Python version"
```bash
# Usa Python 3.11
pyenv install 3.11.9
pyenv shell 3.11.9
```

## 📦 Dependencias Principales

- **pandas** 2.1.4 - Manipulación de datos
- **numpy** 1.26.3 - Cálculos numéricos
- **scikit-learn** 1.4.0 - Preprocesamiento y métricas
- **xgboost** 2.0.3 - Modelo de predicción
- **matplotlib** 3.8.2 - Visualizaciones
- **seaborn** 0.13.1 - Visualizaciones estadísticas

## 💡 Cómo Mejorar el Score (en orden de prioridad)

1. **Ajusta hiperparámetros** en `config.py` (max_depth, learning_rate, etc.)
2. **Añade features simples** en `src/feature_engineering.py` (ej: precio * num_stores)
3. **Prueba diferentes encodings** para categóricas
4. **Revisa feature importance** (`models/feature_importance.csv`)
5. **Si ya funciona bien:** Entonces sí, añade features complejas

## 📚 Información del Datathon

**Objetivo:** Predecir la cantidad de producción óptima para cada prenda

**Métrica de Kaggle:** Score personalizado (0-100) que penaliza más las ventas perdidas que el exceso de stock

**Penalización:** Perder ventas (underproduce) es 2x peor que tener exceso de stock (overproduce)

## 📄 Licencia

MIT License - Ver archivo LICENSE

---

**¿Preguntas?** Consulta el código fuente, está documentado 📖
