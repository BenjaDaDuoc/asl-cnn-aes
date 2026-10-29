# Informe técnico — Evaluación Parcial 2: CNN para letras A, E y S

**Asignatura:** Técnicas Avanzadas de Machine Learning I  
**Proyecto:** Reconocimiento de configuraciones estáticas A/E/S del alfabeto dactilológico estadounidense (ASL)  
**Implementación:** Python, TensorFlow/Keras, scikit-learn  
**Estado:** trabajo experimental completado en `develop`, incluido benchmark controlado de tiempos MLP/CNN; pendientes solamente revisión final, defensa y eventual fusión del repositorio.  
**Evidencias:** [notebook ejecutado](notebooks/proyecto_cnn_aes.ipynb), [reportes](reports/README.md), [modelos](models/), [visualizaciones](images/).

## 1. Descripción del problema de negocio y continuidad con la EP1

El problema abordado consiste en clasificar automáticamente imágenes de **tres configuraciones estáticas de mano**, correspondientes a las letras **A, E y S** del alfabeto dactilológico de ASL. La motivación aplicada es explorar la clasificación de gestos visualmente semejantes, como apoyo académico a técnicas de visión computacional. **No** se desarrolla un traductor completo, ni un sistema validado para asistir a personas sordas, ni un reconocedor de Lengua de Señas Chilena (LSCh).

La variante **A/E/S** se mantiene respecto a la primera evaluación, cuyo modelo fue un **perceptrón multicapa (MLP)**. Conservar las clases, los CSV originales y las 1.075 imágenes de prueba permite observar qué cambia cuando se introduce una red convolucional. Las clases se seleccionaron por sus similitudes de configuración de mano; no se ha demostrado que sean las tres letras intrínsecamente más difíciles de todo el dataset.

## 2. Objetivos y KPIs

**Objetivo general:** entrenar y evaluar una CNN capaz de distinguir A, E y S y compararla con la MLP de la EP1 usando los mismos datos de prueba.

**Objetivos específicos:**
1. Adaptar los registros CSV a tensores `28×28×1` conservando la geometría espacial.
2. Implementar y justificar convoluciones, pooling, capas densas e hiperparámetros.
3. Entrenar dos variantes de CNN, monitorear sus curvas y elegir una **solo por validación**.
4. Medir el test mediante accuracy, precision, recall, F1 y matriz de confusión; interpretar las equivocaciones por clase.
5. Comparar experimentalmente la CNN y la MLP, visualizar Feature Maps y mapas Grad-CAM, y reconocer limitaciones.

| KPI | Umbral propuesto antes de evaluar | CNN en test | Cumple |
|---|---|---:|---|
| Accuracy | > 90 % | **97,95 %** | Sí |
| F1 macro | > 0,85 | **0,9772** | Sí |
| Recall de cada clase | > 0,80 | Mínimo **0,9146 (S)** | Sí |
| Confusión entre letras | Identificar los pares dominantes | **S→E: 21 imágenes** | Identificada |

Estos KPIs se interpretan como metas **del experimento académico**, no compromisos de funcionamiento en condiciones reales.

## 3. Fuentes de datos y análisis exploratorio (EDA)

**Fuente:** [Sign Language MNIST (Kaggle)](https://www.kaggle.com/datasets/datamunge/sign-language-mnist). Se conservaron `data/sign_mnist_train.csv` y `data/sign_mnist_test.csv`. Cada registro incluye la etiqueta original (`label`) y 784 valores `pixel1…pixel784`. Son imágenes de **28×28 píxeles en escala de grises**. El dataset original contiene **24 clases** y 27.455 imágenes en train más 7.172 en test.

**Variante usada:** etiquetas originales **0→A**, **4→E**, **18→S**, remapeadas a **0, 1, 2**.

| Letra | Train original | Test original |
|---|---:|---:|
| A | 1.126 | 331 |
| E | 957 | 498 |
| S | 1.199 | 246 |
| **Total** | **3.282** | **1.075** |

Se comprobó la estructura esperada de las 785 columnas, la ausencia de valores nulos y el rango de píxeles entre 0 y 255. Las visualizaciones de EDA muestran similitudes entre posiciones de dedos, variación de intensidad y ligeros cambios de encuadre. Se verificó que no había coincidencias de imagen **exactamente idénticas** entre las particiones analizadas, pero eso **no descarta casi duplicados** generados a partir de una misma fotografía.

## 4. Preparación de los datos para CNN

Se conservó el test oficial (1.075 imágenes A/E/S) sin usarlo para entrenar ni seleccionar hiperparámetros. Sobre las 3.282 imágenes del train original se aplicó división **estratificada 80/20**, con semilla **42**: **2.625 entrenamiento** y **657 validación**.

La normalización consistió en dividir cada píxel por **255** y convertir a `float32`, obteniendo intensidades entre 0 y 1. Se construyeron tensores `(n,28,28,1)` para la CNN y etiquetas one-hot de tres clases para `categorical_crossentropy`. La MLP, en cambio, recibe imágenes **aplanadas** de 784 valores; esa diferencia permite que la CNN conserve vecindad espacial antes de la etapa de clasificación.

## 5. Metodología CRISP-DM

1. **Comprensión del negocio:** reconocimiento experimental de tres configuraciones A/E/S y definición de KPIs.
2. **Comprensión de datos:** inspección de los CSV, calidad, distribución de clases y visualización de manos.
3. **Preparación:** filtrado A/E/S, mapeo de etiquetas, normalización y particiones train/val/test.
4. **Modelado:** construcción de dos CNN con distinta cantidad de filtros, usando TensorFlow/Keras.
5. **Evaluación:** análisis de curvas, selección por validación, test reservado y comparación con MLP; matriz de confusión, ejemplos y activaciones.
6. **Despliegue:** fuera del alcance; se entregan **notebook reproducible, reportes y modelos guardados**, no un producto puesto en producción.

## 6. Diseño de CNN e hiperparámetros

**Arquitectura general:** entrada `28×28×1` → `Conv2D(3×3, ReLU, same)` → `MaxPooling2D(2×2)` → segunda `Conv2D(3×3, ReLU, same)` → segundo `MaxPooling2D(2×2)` → `Flatten` → `Dense(64, ReLU)` → `Dense(3, softmax)`.

Las convoluciones aprenden filtros locales con pesos compartidos para detectar patrones visuales en diferentes posiciones; ReLU añade no linealidad. Pooling resume regiones y reduce alto/ancho de **28→14→7**, disminuyendo la dimensionalidad espacial. La capa `Flatten` prepara los mapas para la combinación en las capas densas y `softmax` entrega tres probabilidades de clase.

| Variante | Filtros 1 / 2 | Parámetros |
|---|---|---:|
| CNN base | 16 / 32 | 105.411 |
| CNN con más filtros | 32 / 64 | 219.779 |
| MLP EP1 | No utiliza convoluciones | 108.931 |

Se mantuvieron constantes: **Adam** con learning rate inicial **0,001**, `categorical_crossentropy`, **batch 64**, máximo **40 épocas**, **EarlyStopping** (`val_loss`, paciencia 7, recuperar mejores pesos) y **ReduceLROnPlateau** (factor 0,5, paciencia 3). La comparación de dos CNN cambió principalmente la cantidad de filtros para explorar capacidad frente a complejidad; **más filtros no garantizan mayor accuracy**.

## 7. Entrenamiento y selección por validación

Las dos CNN alcanzaron 100 % de accuracy sobre el train y la validación **en este conjunto experimental**. La CNN base ejecutó **35 épocas / 87,44 s**, y la CNN con más filtros **30 épocas / 129,38 s** en la misma sesión de CPU. Se seleccionó **`cnn_mas_filtros` antes de evaluar el test**, porque logró menor `val_loss` con los pesos restaurados: **0,0003516** frente a **0,0005383** de la base.

La pérdida mínima observada en el historial puede ocurrir en una época diferente a la restaurada por EarlyStopping debido a `min_delta=1e-4`; no deben confundirse ambos conceptos. Un 100 % de validación obliga a ser cautelosos: las imágenes pueden incluir transformaciones parecidas de una fotografía original. No prueba generalización a manos de personas desconocidas.

**Evidencias:** [historial CNN base](reports/historial_cnn_base.csv), [historial CNN con más filtros](reports/historial_cnn_mas_filtros.csv), [curvas de validación](images/comparacion_val_loss_cnn.png).

## 8. Evaluación final: accuracy, precision, recall, F1 y matriz

El checkpoint seleccionado se evaluó **sin volver a entrenar** sobre las 1.075 imágenes reservadas de test. La evaluación se reprodujo en Colab con **Keras/TensorFlow** y coincide con los resultados del análisis del checkpoint en Keras con backend PyTorch.

| Métrica global (test) | CNN seleccionada |
|---|---:|
| Accuracy | **97,95 %** |
| Precision macro | **98,50 %** |
| Recall macro | **97,09 %** |
| F1 macro | **97,72 %** |
| Aciertos | **1.053 / 1.075** |
| Errores | **22** |

**Matriz de confusión (filas: clase real / columnas: predicción):**

| Real \\ Pred. | A | E | S |
|---|---:|---:|---:|
| A | 331 | 0 | 0 |
| E | 0 | 497 | 1 |
| S | 0 | **21** | 225 |

Los errores se concentran en **S→E**. El recall de S es **91,46 %** (225/246), frente a **99,80 %** para E y **100 %** para A. La accuracy alta debe interpretarse junto con estas diferencias entre clases. Véase [matriz de confusión](images/matriz_confusion_test_cnn.svg) y [métricas detalladas](reports/metricas_test_cnn.json).

## 9. Comparación experimental MLP vs. CNN

Se cargaron los pesos originales de la MLP de EP1 y los pesos guardados de la CNN; ambos se evaluaron sobre **exactamente las mismas 1.075 imágenes** sin reentrenar ni optimizar con test.

| Métrica (test) | MLP EP1 | CNN EP2 |
|---|---:|---:|
| Accuracy | 94,51 % | **97,95 %** |
| Precision macro | 93,88 % | **98,50 %** |
| Recall macro | 93,51 % | **97,09 %** |
| F1 macro | 93,65 % | **97,72 %** |
| Aciertos | 1.016 | **1.053** |
| Parámetros | 108.931 | 219.779 |

La CNN obtuvo **37 aciertos adicionales**, equivalente a una mejora absoluta de **3,44 puntos porcentuales** de accuracy. Todos los **22 errores** de CNN también aparecen entre los errores de MLP, y la MLP tiene **37 fallos adicionales** corregidos por CNN. La mejora de recall es particularmente visible en S, de **84,96 % a 91,46 %** (+6,50 pp).

**Ventaja CNN:** extracción automática de patrones locales y conservación de relaciones espaciales mediante filtros; **desventajas:** más parámetros en la variante elegida, mayor costo y una interpretación imperfecta. **Ventaja MLP:** arquitectura densa más directa para datos tabulares o vectores aplanados; **desventaja para imágenes:** no utiliza explícitamente la estructura espacial 2D.

**Tiempo de entrenamiento (benchmark homogéneo en Colab):** se entrenaron **tres modelos nuevos** en la misma sesión, con las mismas particiones train/val, batch size 64, Adam, tasa inicial 0,001, 40 épocas máximas y callbacks de EarlyStopping/ReduceLROnPlateau. Se midió el tiempo transcurrido de `fit()`, incluyendo las validaciones por época.

| Modelo reentrenado para benchmark | Épocas efectivas | Tiempo de fit | Parámetros | Accuracy validación |
|---|---:|---:|---:|---:|
| MLP | 40 | **31,92 s** | 108.931 | 100 % |
| CNN base | 35 | **79,13 s** | 105.411 | 100 % |
| CNN más filtros | 30 | **128,02 s** | 219.779 | 100 % |

En esta sesión, la **MLP fue la más rápida** y la CNN elegida tardó unas **4,01 veces** más que la MLP, pese a ejecutar menos épocas (30 frente a 40). Esta comparación representa el **tiempo total bajo la misma política de parada**, no costo por época ni una medición independiente de diferencias entre hardware. Los checkpoints históricos utilizados para las métricas de **test no fueron reemplazados** por los modelos del benchmark; los porcentajes de test anteriores se mantienen. El 100 % de validación de los tres modelos no demuestra generalización externa. Evidencia: [CSV de benchmark](reports/benchmark_tiempos_mlp_cnn.csv) y celda ejecutada de la sección 25 del notebook.

**Evidencia:** [tabla comparativa](reports/comparacion_mlp_cnn_test.csv), [resultados por letra](reports/comparacion_por_letra_mlp_cnn.csv), [gráfico comparativo](images/comparacion_recall_mlp_cnn.png).

## 10. Feature Maps, Grad-CAM y análisis cualitativo

Se visualizaron imágenes A/E/S correctamente clasificadas. En la CNN seleccionada, los Feature Maps de la primera convolución tienen forma `(3,28,28,32)`, mientras los de la segunda tienen forma `(3,14,14,64)`. Los primeros muestran respuestas locales, y los segundos respuestas más agregadas o abstractas después de la primera reducción espacial; esto **no** demuestra que un canal reconozca un dedo específico.

Se generaron mapas **Grad-CAM condicionados por clase A, E y S** para tres imágenes, para visualizar regiones con contribuciones positivas a la puntuación previa a softmax. En la combinación **imagen S / clase A** el mapa no tuvo contribuciones positivas tras ReLU; no debe interpretarse como ausencia absoluta de información visual. Grad-CAM es una herramienta de interpretación aproximada, no prueba causal ni segmentación precisa.

**Evidencias:** [Feature Maps primera capa](images/feature_maps_primera_conv.png), [segunda capa](images/feature_maps_segunda_conv.png), [Grad-CAM por clase](images/gradcam_activaciones_por_clase.png). El notebook también muestra ejemplos correctos e incorrectos del test.

## 11. Limitaciones, ética y posibles mejoras

- **Variantes de origen:** fotografías aumentadas o parecidas pueden contaminar indirectamente una partición aleatoria. Comprobar solo duplicados exactos no resuelve este problema; un test con sujetos o fotografías originales verdaderamente independientes sería más representativo.
- **Alcance:** clasificación de tres signos estáticos ASL, sin reconocer transiciones, expresiones, frases ni LSCh; los resultados no validan uso asistivo real.
- **Representatividad:** ausencia de metadatos sobre personas, diversidad de tonos de piel, iluminación y cámara para cuantificar sesgos. No se deben inventar resultados demográficos.
- **Mejoras futuras:** agrupar muestras por fotografía de origen, ampliar clases y sujetos, introducir augmentación **solo en train**, considerar Dropout/regularización y evaluar con un conjunto externo. El test actual no debe utilizarse para reajustar estos hiperparámetros.

## 12. Conclusiones

La CNN seleccionada resolvió mejor que la MLP el benchmark académico A/E/S, pasando de 94,51 % a **97,95 %** de accuracy y de 93,65 % a **97,72 %** de F1 macro. Su fortaleza es procesar la **estructura espacial** de los píxeles antes de clasificar, aunque utiliza más parámetros que la MLP en este experimento. El principal error sigue siendo **S confundida con E** (21 casos), lo que revela una limitación que no se observa mirando solo la accuracy global.

Las curvas de entrenamiento y la validación perfecta muestran ajuste alto a los datos disponibles, pero **no prueban generalización externa**. Las visualizaciones Feature Maps y Grad-CAM muestran patrones aprendidos y zonas de activación, sin convertirlos en explicaciones causales definitivas. El benchmark de tiempos MLP/CNN confirmó un costo de entrenamiento superior de la CNN seleccionada en esta sesión; la validación con fotos y personas independientes continúa pendiente como trabajo futuro.

## 13. Reproducibilidad y archivos

Clona la rama de desarrollo:

```bash
git clone --branch develop https://github.com/BenjaDaDuoc/asl-cnn-aes.git
cd asl-cnn-aes
python -m pip install -r requirements.txt
jupyter notebook
```

Abre `notebooks/proyecto_cnn_aes.ipynb`. Está organizado en etapas de EDA, arquitectura, entrenamiento, test y comparación. Las celdas de evaluación y visualización de las etapas 4 y 5 pueden reproducirse con los modelos guardados sin reentrenar; las celdas de entrenamiento permiten repetir el experimento si se requiere. El benchmark de tiempos se ejecutó y está guardado; para evitar reentrenamientos accidentales no es necesario ejecutarlo otra vez. El directorio `data/` contiene los CSV, `models/` los checkpoints, `reports/` las métricas y `images/` las evidencias.

**Fuentes de evidencia:** conjunto de datos Sign Language MNIST, código y salidas del notebook, archivos CSV/JSON del proyecto y rúbrica oficial EP2. Los resultados numéricos reportados corresponden a ejecuciones reales documentadas, no a estimaciones.
