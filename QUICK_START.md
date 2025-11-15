# 🚀 Guía Rápida de Inicio

## Instalación en 3 pasos

### 1. Ejecutar setup

```bash
bash setup.sh
```

O manualmente:

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate     # En Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Colocar datos

Coloca los archivos del datathon en la carpeta `data/`:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

### 3. Entrenar y predecir

```bash
# Entrenamiento básico
python train.py

# Con optimización de hiperparámetros (más lento pero mejor)
python train.py --optimize
```

## 📊 Análisis Exploratorio (Opcional)

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## 🔮 Solo Predicciones

Si ya tienes un modelo entrenado:

```bash
python predict.py
```

## 📁 Archivos Generados

Después de entrenar, encontrarás:

- **Modelo**: `models/xgboost_model.json`
- **Submission**: `submissions/submission_YYYYMMDD_HHMMSS.csv`
- **Feature Importance**: `models/feature_importance.csv`

## 🎯 Próximos Pasos

1. Revisa el archivo de submission generado
2. Súbelo a la plataforma del datathon
3. Analiza los resultados
4. Itera mejorando features y parámetros

## 💡 Tips

- **Optimizar hiperparámetros**: Usa `--optimize` para mejor rendimiento (tarda ~1 hora)
- **Revisar feature importance**: Mira `models/feature_importance.csv` para ver qué features son más importantes
- **Experimentar con features**: Modifica `src/feature_engineering.py` para crear nuevas features

## 🆘 Problemas Comunes

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: train.csv"
Asegúrate de que los archivos CSV están en la carpeta `data/`

### Memoria insuficiente
Reduce el tamaño del dataset o ajusta parámetros en `config.py`

## 📖 Documentación Completa

Para más información, consulta [README.md](README.md)

---

**¡Éxito en el datathon! 🥭✨**

