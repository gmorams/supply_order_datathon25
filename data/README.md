# 📊 Carpeta de Datos

Coloca aquí los archivos del datathon:

## Archivos Requeridos

- **train.csv**: Dataset de entrenamiento con datos históricos de 4 temporadas
- **test.csv**: Dataset de test para generar predicciones
- **sample_submission.csv**: Ejemplo del formato de submission

## Estructura Esperada

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Formato de los Datos

### train.csv

Contiene datos históricos con las siguientes columnas principales:

- **ID**: Identificador del modelo
- **id_season**: Identificador de temporada
- **family**: Familia del producto
- **category**: Categoría del producto
- **Production**: Variable target (cantidad a producir)
- **weekly_sales**: Ventas semanales
- **weekly_demand**: Demanda semanal
- Y muchas más features...

### test.csv

Similar a train.csv pero sin la columna `Production` (es lo que debemos predecir).

### sample_submission.csv

Formato de salida esperado:

```csv
ID,Production
1,5000
2,3500
3,7200
...
```

## ⚠️ Importante

- Los archivos de datos NO están incluidos en el repositorio por privacidad
- Descárgalos desde la plataforma del datathon
- Los datos están normalizados (valores entre 0 y 1)

## 📝 Notas

- Los datos son confidenciales de Mango
- No compartir los datos fuera del datathon
- Respetar la licencia y términos de uso

