# Evidencias de entrenamiento — Evaluación Parcial 2 (CNN)

**Asignatura:** Técnicas Avanzadas de Machine Learning I (TLY1102)  
**Dataset:** Sign Language MNIST, variante A/E/S  
**Fecha del entrenamiento:** 29 de octubre de 2026 (UTC indicado en resultados_etapa3_cnn.json)  
**Entorno registrado:** Python 3.13.18, TensorFlow 2.21.0, sin GPU detectada.

## Protocolo

Se entrenaron las arquitecturas `cnn_base` (16/32 filtros) y `cnn_mas_filtros` (32/64 filtros) con:
- 2.625 imágenes de entrenamiento y 657 de validación.
- Entrada (28,28,1), Adam con learning rate inicial de 0,001 y `categorical_crossentropy`.
- Batch size de 64, hasta 40 épocas, `EarlyStopping` con paciencia 7 y restauración de mejores pesos, y `ReduceLROnPlateau` con paciencia 3.
- Selección mediante pérdida **de validación**; el test oficial de 1.075 imágenes permanece reservado.

## Resultados de validación (pesos restaurados)

| Medida | CNN base | CNN más filtros |
|---|---:|---:|
| Parámetros entrenables | 105.411 | 219.779 |
| Épocas ejecutadas | 35 | 30 |
| Tiempo fit en CPU (s) | 87,44 | 129,38 |
| Accuracy entrenamiento | 100 % | 100 % |
| Accuracy validación | 100 % | 100 % |
| Pérdida validación del modelo restaurado | 0,0005383 | 0,0003516 |
| Mínima pérdida de validación observada en el historial | 0,0004426 | 0,0002740 |

**Selección programada:** `cnn_mas_filtros` por su menor pérdida de validación de los pesos restaurados. Esto **no demuestra** mejor accuracy, porque ambos modelos obtuvieron 100 % en el conjunto de validación y la CNN seleccionada necesitó más tiempo y parámetros.

**Nota técnica sobre EarlyStopping:** la época de mínima pérdida observada y la época de pesos restaurados no coinciden necesariamente, debido a `min_delta=1e-4`. Los campos `mejor_epoca_val_loss` del JSON/CSV se calculan con `argmin(val_loss)`, no con la época de pesos efectivamente restaurados. Se interpretará esta diferencia explícitamente en el informe.

## Limitaciones e interpretación

La validación es aleatoria estratificada sobre imágenes derivadas de fotografías de origen. Es posible que variaciones muy parecidas de una misma fotografía estén presentes en entrenamiento y validación. **Una validación del 100 % no prueba generalización** a fotografías de origen desconocidas, personas nuevas, iluminación distinta ni uso real. La comparación formal con el MLP anterior debe utilizar las mismas particiones y, para contrastar tiempos, el mismo equipo y condiciones.

**Archivos fuente:** `historial_cnn_base.csv`, `historial_cnn_mas_filtros.csv`, `resumen_entrenamiento_cnn.csv`, `resultados_etapa3_cnn.json` y salidas integradas en `notebooks/proyecto_cnn_aes.ipynb`. Los archivos de pesos y gráficos se generan por separado y deben conservarse en `models/` e `images/`, respectivamente.

**Pendiente:** comprobar generalización con el test reservado en la etapa 4, calcular accuracy/precision/recall/F1, matriz de confusión, ejemplos de aciertos y errores, y luego Feature Maps.
