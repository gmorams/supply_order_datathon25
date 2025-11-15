#!/bin/bash

# Script de configuración para el proyecto Mango Datathon

echo "🥭 Configurando proyecto Mango Datathon..."
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado"
    exit 1
fi

echo "✅ Python encontrado: $(python3 --version)"
echo ""

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
python3 -m venv venv

# Activar entorno virtual
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "✅ Entorno virtual creado y activado"
echo ""

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Dependencias instaladas"
echo ""

# Crear directorios si no existen
echo "📁 Creando estructura de directorios..."
mkdir -p data models submissions notebooks

echo "✅ Directorios creados"
echo ""

# Mensaje final
echo "🎉 ¡Configuración completada!"
echo ""
echo "📋 Próximos pasos:"
echo "   1. Activar el entorno virtual:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "      source venv/Scripts/activate"
else
    echo "      source venv/bin/activate"
fi
echo "   2. Coloca tus datos en la carpeta 'data/'"
echo "   3. Ejecuta: python train.py"
echo ""
echo "🚀 ¡Buena suerte en el datathon!"

