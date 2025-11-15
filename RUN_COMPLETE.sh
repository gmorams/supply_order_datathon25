#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║        🥭  MANGO DATATHON - EJECUCIÓN COMPLETA  🥭              ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar que estamos en el directorio correcto
cd "$(dirname "$0")"

echo "📍 Directorio actual: $(pwd)"
echo ""

# Paso 1: Verificar archivos de datos
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  VERIFICANDO ARCHIVOS DE DATOS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

missing_files=0

if [ -f "data/train.csv" ]; then
    echo -e "${GREEN}✅${NC} train.csv encontrado"
    echo "   Tamaño: $(ls -lh data/train.csv | awk '{print $5}')"
else
    echo -e "${RED}❌${NC} train.csv NO encontrado"
    missing_files=$((missing_files + 1))
fi

if [ -f "data/test.csv" ]; then
    echo -e "${GREEN}✅${NC} test.csv encontrado"
    echo "   Tamaño: $(ls -lh data/test.csv | awk '{print $5}')"
else
    echo -e "${RED}❌${NC} test.csv NO encontrado"
    missing_files=$((missing_files + 1))
fi

if [ -f "data/sample_submission.csv" ]; then
    echo -e "${GREEN}✅${NC} sample_submission.csv encontrado"
    echo "   Tamaño: $(ls -lh data/sample_submission.csv | awk '{print $5}')"
else
    echo -e "${RED}❌${NC} sample_submission.csv NO encontrado"
    missing_files=$((missing_files + 1))
fi

echo ""

if [ $missing_files -gt 0 ]; then
    echo -e "${RED}❌ ERROR: Faltan $missing_files archivo(s)${NC}"
    echo ""
    echo "Por favor, coloca los siguientes archivos en la carpeta data/:"
    echo "  • train.csv"
    echo "  • test.csv"
    echo "  • sample_submission.csv"
    echo ""
    echo "Ruta completa: $(pwd)/data/"
    exit 1
fi

# Paso 2: Verificar entorno virtual
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  CONFIGURANDO ENTORNO PYTHON"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚙️${NC}  Creando entorno virtual..."
    python3 -m venv venv
    echo -e "${GREEN}✅${NC} Entorno virtual creado"
else
    echo -e "${GREEN}✅${NC} Entorno virtual ya existe"
fi

echo ""
echo -e "${YELLOW}⚙️${NC}  Activando entorno virtual..."
source venv/bin/activate

echo -e "${GREEN}✅${NC} Entorno virtual activado"
echo ""

# Paso 3: Instalar dependencias
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  INSTALANDO DEPENDENCIAS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${YELLOW}⚙️${NC}  Actualizando pip..."
pip install --upgrade pip --quiet

echo -e "${YELLOW}⚙️${NC}  Instalando librerías (puede tardar unos minutos)..."
pip install -r requirements.txt --quiet

echo -e "${GREEN}✅${NC} Dependencias instaladas"
echo ""

# Paso 4: Preguntar modo de entrenamiento
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  SELECCIONAR MODO DE ENTRENAMIENTO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Elige el modo de entrenamiento:"
echo ""
echo "  1) RÁPIDO (5-10 min) - Usa parámetros por defecto"
echo "  2) OPTIMIZADO (1-2 horas) - Optimiza hiperparámetros con Optuna"
echo ""
read -p "Selecciona una opción (1 o 2): " option
echo ""

# Paso 5: Entrenar modelo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  ENTRENANDO MODELO XGBOOST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$option" == "2" ]; then
    echo -e "${YELLOW}🚀${NC} Iniciando entrenamiento OPTIMIZADO..."
    echo -e "${YELLOW}⏱️${NC}  Esto puede tardar 1-2 horas..."
    echo ""
    python train.py --optimize
else
    echo -e "${YELLOW}🚀${NC} Iniciando entrenamiento RÁPIDO..."
    echo -e "${YELLOW}⏱️${NC}  Esto tomará 5-10 minutos..."
    echo ""
    python train.py
fi

# Verificar si el entrenamiento fue exitoso
if [ $? -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    echo "║                    ✅  ¡ÉXITO COMPLETO!  ✅                      ║"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Archivos generados:"
    echo ""
    echo "   🤖 Modelo entrenado:"
    echo "      └─ models/xgboost_model.json"
    echo ""
    echo "   📈 Feature importance:"
    echo "      └─ models/feature_importance.csv"
    echo ""
    echo "   📤 Archivo de submission:"
    latest_submission=$(ls -t submissions/*.csv 2>/dev/null | head -1)
    if [ -n "$latest_submission" ]; then
        echo "      └─ $latest_submission"
    fi
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 PRÓXIMOS PASOS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. Revisa el archivo de submission en submissions/"
    echo "2. Sube el archivo a la plataforma del datathon"
    echo "3. ¡Espera tus resultados! 🏆"
    echo ""
    echo "Para ver la importancia de features:"
    echo "  cat models/feature_importance.csv | head -20"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Error durante el entrenamiento${NC}"
    echo ""
    echo "Revisa los mensajes de error arriba y verifica:"
    echo "  • Los archivos CSV están correctos"
    echo "  • Tienes suficiente memoria RAM"
    echo "  • Las dependencias están instaladas"
fi

