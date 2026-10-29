# EP2 — CNN de reconocimiento de letras A, E y S

**Informe técnico consolidado para la entrega:** [INFORME_TECNICO_EP2.md](INFORME_TECNICO_EP2.md) (problema, objetivos, KPIs, EDA, CRISP-DM, arquitectura, resultados, errores, activaciones, conclusiones y reproducibilidad). Este README registra la historia de desarrollo por etapas.

**Asignatura:** Técnicas Avanzadas de Machine Learning I (TLY1102)  
**Problema:** clasificación de imágenes de signos estáticos del alfabeto dactilológico estadounidense (ASL).  
**Variante del equipo:** letras **A, E y S**, igual que en la Evaluación Parcial 1 (EP1).

**Estado:** etapas 1–5 ejecutadas en Colab y guardadas en la rama `develop`, incluyendo comparación MLP/CNN, Feature Maps y Grad-CAM para cada clase. Los seis archivos externos de la etapa 5 están subidos a `images/` y `reports/`. El benchmark controlado de tiempos MLP/CNN ya se ejecutó en la misma sesión de Colab; resultados guardados en `reports/benchmark_tiempos_mlp_cnn.csv` (MLP 31,92 s; CNN base 79,13 s; CNN seleccionada 128,02 s). Este documento acompaña el trabajo en `develop`; el README original de la MLP se conserva en esta etapa.

## Archivo principal

- `notebooks/proyecto_cnn_aes.ipynb`: notebook documentado de la segunda evaluación.
- `notebooks/proyecto_lenguaje_señas.ipynb`: MLP original de la EP1; sirve como línea base.
- `data/sign_mnist_train.csv`, `data/sign_mnist_test.csv`: dataset original ya incluido en el repositorio.
- `models/mlp_asl_aes.keras`: modelo anterior; **no se sobrescribe**.
- `models/cnn_base.keras` y `models/cnn_mas_filtros.keras`: ambos modelos CNN entrenados, guardados en `develop`.
- `requirements.txt`: dependencias del proyecto (incluye TensorFlow/Keras).
- `reports/`: historiales completos, resumen CSV/JSON y [análisis de resultados](reports/README.md) de la etapa 3.

## Comprobaciones de los datos

| Característica | Valor |
|---|---:|
| Imágenes originales de entrenamiento | 27.455 |
| Imágenes originales de prueba | 7.172 |
| Resolución | 28 × 28 × 1 |
| Intensidad original | 0–255 |
| Clases originales | 24 |
| Clases de la variante | A (0), E (4), S (18) |
| Imágenes A/E/S de train original | 3.282 |
| Imágenes A/E/S de test original | 1.075 |
| Entrenamiento tras división estratificada | 2.625 |
| Validación tras división estratificada | 657 |
| Test reservado | 1.075 |
| Forma de entrada a la CNN | (28, 28, 1) |
| Normalización | píxel / 255 |
| Etiquetas nuevas | A→0, E→1, S→2 |

Las imágenes fueron repartidas usando `train_test_split(..., test_size=0.20, stratify=y, random_state=42)`, solo sobre el conjunto de entrenamiento original. La división conserva el test oficial sin usarlo para entrenar ni seleccionar el modelo.

**Auditoría preliminar:** no se detectaron duplicados *exactos* entre train/val/test en los datos A/E/S. Esto **no descarta** fotografías casi idénticas producidas por aumentos del dataset. Para extraer conclusiones fiables sobre generalización hay que analizar ese riesgo y mantener un protocolo comparable al reentrenar MLP/CNN.

## Ejecución local

1. Clona el repositorio y cambia a la rama `develop`:
   ```bash
   git clone --branch develop https://github.com/BenjaDaDuoc/asl-cnn-aes.git
   cd asl-cnn-aes
   ```
2. Instala dependencias:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Abre Jupyter:
   ```bash
   jupyter notebook
   ```
4. Abre `notebooks/proyecto_cnn_aes.ipynb` y selecciona **Run All**. El notebook detecta los CSV automáticamente si se ejecuta desde la raíz del repositorio o desde `notebooks/`.

Para Google Colab, clona el repositorio en `/content/asl-cnn-aes` y abre el notebook desde esa carpeta; el código también busca los CSV en esa ubicación. No es necesario cargar el ZIP dos veces.

## Métricas base de EP1 (referencia histórica, no resultado de CNN)

| Métrica MLP (test) | Valor |
|---|---:|
| Accuracy | 0,9451 |
| Precision macro | 0,9388 |
| Recall macro | 0,9351 |
| F1 macro | 0,9365 |
| Parámetros | 108.931 |
| Aciertos | 1.016 / 1.075 |

La validación anterior alcanzó un resultado demasiado alto para asumir que mide generalización a fotografías originales desconocidas. Cuando se compare rigurosamente con la CNN, habrá que reejecutar la MLP con particiones y condiciones equivalentes; **no se compararán tiempos de entrenamiento sin medirlos bajo las mismas condiciones**.

## Arquitecturas propuestas (etapa 2)

Las dos alternativas reciben imágenes de (28, 28, 1) y aplican dos bloques `Conv2D(3×3, ReLU, padding=same) → MaxPooling2D(2×2)`, seguidos por `Flatten → Dense(64, ReLU) → Dense(3, softmax)`.

| Configuración | Filtros convolucionales | Parámetros entrenables |
|---|---|---:|
| `cnn_base` | 16 → 32 | 105.411 |
| `cnn_mas_filtros` | 32 → 64 | 219.779 |
| MLP EP1 (solo referencia histórica) | No aplica | 108.931 |

Ambas CNN usan Adam (learning rate 0,001) y pérdida `categorical_crossentropy` con etiquetas one-hot. Se utilizó batch de 64 y un máximo de 40 épocas con EarlyStopping; el entrenamiento ya se ejecutó. Los parámetros son conteos estructurales, no resultados de predicción. La alternativa con más filtros permite estudiar el costo y efecto de incrementar capacidad sin modificar simultáneamente otros hiperparámetros.

El notebook imprime `model.summary()`, compara parámetros y comprueba la forma de salida y la normalización softmax con una pequeña muestra de **validación**, no de test. Esas probabilidades todavía no tienen significado predictivo.

## Etapa 3 — Entrenamiento y validación

**El código se ejecutó con TensorFlow en Colab (29-10-2026)**. El notebook:

1. Construye nuevamente desde cero `cnn_base` y `cnn_mas_filtros` con semilla 42.
2. Entrena hasta 40 épocas por modelo y batch size 64; aplica **EarlyStopping** (`val_loss`, paciencia 7, restauración de mejores pesos) y **ReduceLROnPlateau** (`val_loss`, factor 0,5, paciencia 3).
3. Guarda automáticamente `models/cnn_base.keras` y `models/cnn_mas_filtros.keras`, historiales en `reports/`, curvas en `images/` y un JSON con resultados, hiperparámetros, tiempos y entorno.
4. Elige el modelo con **menor val_loss** con los pesos restaurados (solo validación) y genera un diagnóstico con valores reales de entrenamiento, posible sobreajuste y alertas de validación casi perfecta.

Los **resultados reales de validación** están registrados en `reports/README.md` y en los CSV/JSON. La CNN base alcanzó 100 % de accuracy en validación tras 35 épocas (87,44 s) y la CNN con más filtros también 100 % tras 30 épocas (129,38 s); fue seleccionada por menor `val_loss` del modelo restaurado. Esto **no demuestra generalización externa**. El test original permanece reservado para la etapa 4. La validación aleatoria tiene riesgo de contar variaciones de una fotografía de origen en train y val, por lo que puede ser demasiado optimista.

### Reproducción del entrenamiento en Google Colab

1. Abre `notebooks/proyecto_cnn_aes.ipynb` desde la rama `develop` en Google Colab.
2. Ejecuta **antes de todas las celdas**, en una celda temporal:
   ```python
   !git clone --branch develop https://github.com/BenjaDaDuoc/asl-cnn-aes.git /content/asl-cnn-aes
   ```
3. En caso de que falten paquetes, ejecuta `!pip install -r /content/asl-cnn-aes/requirements.txt` y reinicia el entorno de ejecución. Luego usa **Entorno de ejecución → Ejecutar todas**.
4. Los archivos se guardan en `/content/asl-cnn-aes/models/`, `/content/asl-cnn-aes/reports/` y `/content/asl-cnn-aes/images/`. **Descárgalos o súbelos a GitHub antes de cerrar Colab**, porque su almacenamiento es temporal.

### Archivos producidos por la ejecución

- `models/cnn_base.keras` y `models/cnn_mas_filtros.keras`.
- `reports/historial_cnn_base.csv`, `reports/historial_cnn_mas_filtros.csv`, `reports/resumen_entrenamiento_cnn.csv`, `reports/resultados_etapa3_cnn.json`.
- `images/loss_cnn_base.png`, `images/accuracy_cnn_base.png`, `images/loss_cnn_mas_filtros.png`, `images/accuracy_cnn_mas_filtros.png`, `images/comparacion_val_loss_cnn.png`.

## Etapa 4 — Evaluación del modelo seleccionado (test oficial)

**Modelo fijado antes del test:** `cnn_mas_filtros`, seleccionado por menor `val_loss` restaurada durante la etapa 3.

La evaluación realizada con el checkpoint entrenado, cargado en Keras 3.13.2 (backend PyTorch), utilizó **1.075 imágenes del test original**. El modelo no se reentrenó ni se ajustaron sus hiperparámetros.

| Métrica | Resultado de test |
|---|---:|
| Accuracy | **97,95 %** (1.053/1.075) |
| Precision macro | 98,50 % |
| Recall macro | 97,09 % |
| F1 macro | 97,72 % |
| Errores | 22 |

**Matriz de confusión**, filas reales / columnas predichas:

| Real \\ Predicción | A | E | S |
|---|---:|---:|---:|
| A | 331 | 0 | 0 |
| E | 0 | 497 | 1 |
| S | 0 | **21** | 225 |

La dificultad principal fue **S confundida con E** (21 imágenes, recall de S ≈ 91,46 %). La MLP previa había alcanzado 94,51 % de accuracy: la CNN mejora en **3,44 puntos porcentuales**, con 37 aciertos adicionales. Es una comparación descriptiva del mismo test; no compara tiempos en las mismas condiciones.

**Evidencias:** [reportes de etapa 4](reports/README.md), [métricas JSON](reports/metricas_test_cnn.json), [matriz en SVG](images/matriz_confusion_test_cnn.svg) y secciones 16–20 del [notebook actualizado](notebooks/proyecto_cnn_aes.ipynb).

**Comprobación de Colab:** las cinco celdas de código de la etapa 4 se ejecutaron en Colab/Keras con backend TensorFlow sin errores y reprodujeron la exactitud de 97,95 % y 22 errores. Las gráficas aparecen en el notebook guardado en GitHub.

**Limitaciones:** el dataset puede compartir variantes de fotografías de origen entre particiones. La accuracy en este benchmark no demuestra desempeño con sujetos nuevos ni valida un traductor de LSCh.

## Etapa 5 — Interpretabilidad y comparación MLP/CNN (ejecutada y verificada)

**Compatibilidad del checkpoint MLP de la EP1:** el archivo `models/mlp_asl_aes.keras` fue serializado con parámetros de `GlorotUniform` que algunas versiones actuales de Keras no admiten (`input_axes`, `output_axes`). La sección 21 se actualizó para reconstruir la arquitectura exacta (784→128→64→3, 108.931 parámetros) y llamar a `load_weights(..., skip_mismatch=False)` en el archivo original. **Se preservan los pesos aprendidos; no se reentrena ni se modifica el modelo original.** La corrección se ejecutó satisfactoriamente en Colab, sin repetir el entrenamiento. Se verificó el mecanismo con una prueba de compatibilidad que reproduce el error de deserialización y recupera las predicciones originales idénticas.

El notebook `notebooks/proyecto_cnn_aes.ipynb` incluye ahora las secciones **21–26**, que permiten analizar por qué mejora la CNN sin alterar su entrenamiento:

1. Carga de `models/mlp_asl_aes.keras` y evaluación del checkpoint MLP sobre **exactamente el mismo test A/E/S**. Las métricas reproducidas coinciden con la EP1: 1.016 aciertos de 1.075.
2. Tabla de accuracy, precision, recall, F1 macro, parámetros entrenables y aciertos por modelo; comparación de recall y errores por clase.
3. Visualización de **Feature Maps** de `conv_bordes` (28×28) y `conv_patrones` (14×14) con ocho filtros de cada capa para una imagen A, E y S.
4. Visualización de **Grad-CAM para cada clase A/E/S** en esas tres imágenes. El cálculo utiliza gradientes del logit previo a softmax con los pesos guardados y no modifica el modelo.
5. **Benchmark controlado ejecutado** con MLP y ambas CNN bajo las mismas condiciones; `EJECUTAR_BENCHMARK_TIEMPOS = True` figura en la ejecución guardada. Se midieron 31,92 s (MLP), 79,13 s (CNN base) y 128,02 s (CNN seleccionada). Estas son mediciones de `fit()` de modelos nuevos, diferentes de los checkpoints seleccionados para test. No es necesario repetir la ejecución.

**Ejecución comprobada:** las cinco celdas de código de etapa 5 guardan salidas sin errores. El checkpoint de MLP reprodujo 94,51 % de accuracy y el de CNN 97,95 %; ambas se evaluaron con las mismas 1.075 imágenes de test. Los errores de CNN son 22 y coinciden con errores de la MLP; la MLP se equivoca en otros 37 ejemplos que la CNN acierta. Los Feature Maps de A/E/S y los nueve mapas Grad-CAM se generaron. Un mapa de clase A para una imagen S no presentó contribuciones positivas (Grad-CAM con ReLU): esto se informó como limitación interpretativa, no como error de ejecución. Los gráficos y CSV se subieron al repositorio.

Las activaciones son interpretaciones aproximadas de patrones visuales, no una prueba de que cada filtro represente un dedo concreto ni de que el modelo funcione con personas nuevas.

## Próximas etapas

1. Revisar las conclusiones finales, las limitaciones del dataset y los criterios de la pauta EP2.
2. Interpretar el benchmark ya ejecutado (MLP 31,92 s; CNN base 79,13 s; CNN más filtros 128,02 s) en la presentación; no mezclar estos tiempos con los del entrenamiento original.
3. Preparar y ensayar la presentación de hasta diez minutos, respaldada en los resultados reales.

## Alcance responsable

No es un sistema para uso real con personas sordas, ni reconoce secuencias dinámicas de señas. En particular, clasificar signos estáticos de ASL no equivale a traducir Lengua de Señas Chilena (LSCh). La revisión de errores por iluminación, encuadre y diversidad de manos es parte de la evaluación.
