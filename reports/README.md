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

## Etapa 4 — Prueba reservada (modelo seleccionado)

Se evaluó exclusivamente el checkpoint `models/cnn_mas_filtros.keras` — elegido antes del test por `val_loss` — sobre las **1.075 imágenes originales de prueba** de A, E y S. El modelo no fue reentrenado ni modificado para realizar esta medición.

La evaluación se ejecutó con **Keras 3.13.2 / backend PyTorch**, cargando los pesos guardados del entrenamiento original en TensorFlow 2.21.0. El notebook incluye celdas que permiten reproducir las métricas directamente en Colab con TensorFlow para la entrega final.

| Métrica de test | CNN seleccionada |
|---|---:|
| Accuracy | **97,95 %** (1.053/1.075) |
| Precision macro | 98,50 % |
| Recall macro | 97,09 % |
| F1 macro | 97,72 % |
| Log loss | 0,04132 |
| Errores | 22 |

### Métricas y errores por letra

| Letra | Precision | Recall | F1 | Imágenes |
|---|---:|---:|---:|---:|
| A | 100,00 % | 100,00 % | 100,00 % | 331 |
| E | 95,95 % | 99,80 % | 97,83 % | 498 |
| S | 99,56 % | **91,46 %** | 95,34 % | 246 |

**Matriz de confusión:** filas = letra real, columnas = predicción.

| Real \\ Predicción | A | E | S |
|---|---:|---:|---:|
| A | 331 | 0 | 0 |
| E | 0 | 497 | 1 |
| S | 0 | **21** | 225 |

La CNN clasificó todas las imágenes A correctamente. La principal dificultad sigue siendo **S → E**: 21 de las 246 imágenes S fueron confundidas con E; solo una imagen E fue clasificada como S. La exactitud global, por sí sola, habría ocultado esa diferencia por clase.

### Comparación descriptiva con la MLP anterior

| Modelo | Accuracy test | Aciertos | F1 macro |
|---|---:|---:|---:|
| MLP de EP1 (histórico) | 94,51 % | 1.016/1.075 | 93,65 % |
| CNN seleccionada de EP2 | **97,95 %** | **1.053/1.075** | **97,72 %** |

La CNN mejoró el porcentaje de aciertos en **3,44 puntos porcentuales** (37 aciertos adicionales), en el mismo conjunto test A/E/S. Esto es evidencia favorable, pero el dataset tiene imágenes transformadas a partir de fotos de origen, por lo que no se extrapola directamente a señas de nuevos sujetos. Las velocidades de entrenamiento **no se comparan** con la MLP histórica.

La etapa 4 queda registrada en `metricas_test_cnn.json`, `metricas_test_cnn.csv`, `metricas_por_clase_test_cnn.csv` y `matriz_confusion_test_cnn.csv`. El notebook `notebooks/proyecto_cnn_aes.ipynb` contiene las celdas para reproducir estos resultados y exportar las imágenes de la matriz de confusión y ejemplos correctos/incorrectos. Las celdas de la etapa 4 **se ejecutaron y sus salidas se guardaron desde Colab con backend TensorFlow**: se confirmaron 1.053 aciertos y 22 errores, sin errores de ejecución. Los gráficos ya aparecen incrustados en el notebook.

**Pendiente para etapa 5:** análisis de activaciones, Feature Maps, comparación experimental ampliada y conclusiones para la presentación oral.

## Etapa 5 — Comparación e interpretabilidad (implementada, pendiente de ejecución)

El notebook actualizado contiene las secciones 21–26 con:

- Carga del `models/mlp_asl_aes.keras` original y evaluación de MLP sobre las mismas 1.075 imágenes, sin modificar los modelos.
- Comparación de accuracy, precision, recall y F1 macro; parámetros y métricas por clase con tablas y gráficos.
- Feature Maps (primeras dos convoluciones, 8 filtros por capa) sobre tres imágenes A/E/S correctamente clasificadas.
- Grad-CAM con los pesos del clasificador final, para las activaciones asociadas a **cada clase** en tres imágenes representativas; no altera los pesos ni selecciona hiperparámetros con test.
- Benchmark controlado opcional de tiempos de entrenamiento MLP vs dos CNN; **desactivado por defecto** para no repetir entrenamiento sin necesidad.
- Interpretación, limitaciones, reflexiones y guía de defensa de máximo 10 minutos.

**Se requiere ejecutar las nuevas celdas 21–26 en Colab** para generar los gráficos y reportes adicionales. Si el Colab actual sigue activo con variables de la etapa 4, no hace falta repetir etapas previas; si no, ejecuta solo las celdas de la etapa 4 para cargar el modelo y el test. Después guarda el notebook ejecutado con Ctrl+S a `develop`. Los PNG/CSV generados se suben por separado.
