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

**Completado:** evaluación del test reservado en la etapa 4 y Feature Maps/Grad-CAM en etapa 5. Queda por evaluar la generalización con fotografías y sujetos independientes, fuera del alcance de la EP2.

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

**Etapa 5 completada:** comparación del checkpoint MLP con CNN y visualizaciones de activaciones registradas en el notebook y en las carpetas de reportes e imágenes.

## Etapa 5 — Comparación e interpretabilidad (ejecutada)

El notebook actualizado contiene las secciones 21–26 con:

- Carga del `models/mlp_asl_aes.keras` original y evaluación de MLP sobre las mismas 1.075 imágenes, sin modificar los modelos.
- Comparación de accuracy, precision, recall y F1 macro; parámetros y métricas por clase con tablas y gráficos.
- Feature Maps (primeras dos convoluciones, 8 filtros por capa) sobre tres imágenes A/E/S correctamente clasificadas.
- Grad-CAM con los pesos del clasificador final, para las activaciones asociadas a **cada clase** en tres imágenes representativas; no altera los pesos ni selecciona hiperparámetros con test.
- Benchmark controlado de tiempos de entrenamiento MLP vs dos CNN; **ejecutado en Colab** y respaldado en el CSV de reportes.
- Interpretación, limitaciones, reflexiones y guía de defensa de máximo 10 minutos.

**Verificado en el notebook guardado:** las cinco celdas de código de esta etapa se ejecutaron sin errores. La MLP original reconstruida a partir de los pesos del checkpoint produjo exactamente 1.016/1.075 aciertos (94,51 %), mientras que la CNN produjo 1.053/1.075 (97,95 %). Los 22 errores de la CNN también fueron errores de la MLP; la MLP cometió 37 errores adicionales que la CNN acertó.

**Comparación por letra:** el recall de E pasó de 95,58 % a 99,80 % (+4,22 pp), y el de S de 84,96 % a 91,46 % (+6,50 pp). A mantuvo 100 % de recall. Los `Feature Maps` generados contienen las formas (3,28,28,32) y (3,14,14,64). Los mapas Grad-CAM para A/E/S se generaron; se informó que una combinación (imagen S, clase A) no mostró contribuciones positivas, algo que no debe interpretarse como ausencia de información visual.

Los archivos `comparacion_mlp_cnn_test.csv`, `comparacion_por_letra_mlp_cnn.csv`, `comparacion_recall_mlp_cnn.png`, `feature_maps_primera_conv.png`, `feature_maps_segunda_conv.png` y `gradcam_activaciones_por_clase.png` ya están subidos a `develop`. El benchmark de tiempos quedó ejecutado y documentado en `benchmark_tiempos_mlp_cnn.csv`. La comparación mide fit total de una misma sesión, no las ejecuciones históricas.

## Benchmark controlado de tiempos — criterio de comparación (completado)

Se reconstruyeron y entrenaron **tres modelos nuevos** con datos, particiones train/validación, semilla, optimizador, batch, máximo de épocas y reglas de detención equivalentes, sin reemplazar los checkpoints usados para el test.

| Modelo | Épocas ejecutadas | Tiempo fit | Parámetros | Accuracy validación |
|---|---:|---:|---:|---:|
| MLP | 40 | 31,92 s | 108.931 | 100 % |
| CNN base | 35 | 79,13 s | 105.411 | 100 % |
| CNN con más filtros | 30 | 128,02 s | 219.779 | 100 % |

La MLP fue **4,01 veces más rápida** que la CNN con más filtros en el tiempo de fit medido bajo esta política de entrenamiento; el ratio es **128,02 / 31,92 ≈ 4,01**. Las épocas efectivas difieren por EarlyStopping y la comparación evalúa **tiempo total** bajo las mismas reglas, no costo por época. El resultado valida la importancia del costo computacional, pero no demuestra generalización a personas nuevas. Datos completos: [benchmark_tiempos_mlp_cnn.csv](benchmark_tiempos_mlp_cnn.csv).
