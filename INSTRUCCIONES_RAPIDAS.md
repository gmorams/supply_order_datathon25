# 🚀 INSTRUCCIONES RÁPIDAS - Mango Datathon

## ¿Dónde colocar los archivos CSV?

### 📁 Coloca tus 3 archivos en la carpeta `data/`:

```
supply_order_datathon25/
└── data/
    ├── train.csv              ⬅️ COLOCA AQUÍ
    ├── test.csv               ⬅️ COLOCA AQUÍ
    └── sample_submission.csv  ⬅️ COLOCA AQUÍ
```

**Ruta completa:**
```
/Users/hugonienhausen/Desktop/datathon/supply_order_datathon25/data/
```

---

## 🎯 Opción 1: Ejecutar TODO Automáticamente (RECOMENDADO)

Abre tu terminal y ejecuta:

```bash
cd /Users/hugonienhausen/Desktop/datathon/supply_order_datathon25
bash RUN_COMPLETE.sh
```

Este script hace TODO por ti:
- ✅ Verifica que los archivos CSV están en su lugar
- ✅ Crea el entorno virtual de Python
- ✅ Instala todas las dependencias
- ✅ Entrena el modelo XGBoost
- ✅ Genera el archivo de submission

**Tiempo estimado:** 5-10 minutos (modo rápido) o 1-2 horas (modo optimizado)

---

## 🎯 Opción 2: Paso a Paso Manual

Si prefieres más control, ejecuta estos comandos uno por uno:

### Paso 1: Ir al directorio del proyecto
```bash
cd /Users/hugonienhausen/Desktop/datathon/supply_order_datathon25
```

### Paso 2: Crear entorno virtual (solo primera vez)
```bash
python3 -m venv venv
```

### Paso 3: Activar entorno virtual
```bash
source venv/bin/activate
```

### Paso 4: Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 5: Entrenar el modelo

**Modo RÁPIDO** (5-10 minutos):
```bash
python train.py
```

**Modo OPTIMIZADO** (1-2 horas, mejor performance):
```bash
python train.py --optimize
```

---

## 📤 ¿Dónde encontrar el archivo de submission?

Después del entrenamiento, tu archivo estará en:

```
submissions/submission_YYYYMMDD_HHMMSS.csv
```

**Este es el archivo que debes subir a la plataforma del datathon.**

---

## 📊 Verificar que los archivos están en su lugar

```bash
ls -la data/
```

Deberías ver:
```
train.csv
test.csv
sample_submission.csv
```

---

## 🐛 Solución de Problemas

### "No such file: train.csv"
➡️ **Solución:** Coloca los archivos CSV en la carpeta `data/`

### "ModuleNotFoundError"
➡️ **Solución:** Ejecuta `pip install -r requirements.txt`

### "Permission denied"
➡️ **Solución:** Ejecuta `chmod +x RUN_COMPLETE.sh`

### El script tarda mucho
➡️ **Normal:** El entrenamiento puede tardar 5-10 minutos (modo rápido) o hasta 2 horas (modo optimizado)

---

## 💡 Comandos Útiles

**Ver las primeras 20 features más importantes:**
```bash
cat models/feature_importance.csv | head -20
```

**Ver estadísticas del submission:**
```bash
python -c "import pandas as pd; df=pd.read_csv('$(ls -t submissions/*.csv | head -1)'); print(df['Production'].describe())"
```

**Solo generar predicciones (si ya entrenaste):**
```bash
python predict.py
```

---

## 📞 ¿Necesitas Ayuda?

1. Revisa el `README.md` para documentación completa
2. Mira `PROJECT_SUMMARY.md` para detalles del proyecto
3. Abre un issue en el repositorio de GitHub

---

## ✅ Checklist Rápido

- [ ] Archivos CSV en la carpeta `data/`
- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas
- [ ] Modelo entrenado exitosamente
- [ ] Archivo de submission generado
- [ ] Submission subido a la plataforma

---

**¡Éxito en el datathon! 🥭🏆**

