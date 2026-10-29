> **EP2 (CNN):** Este repositorio contiene la segunda evaluación en desarrollo. Consulta [README_EP2.md](README_EP2.md) y el [notebook CNN](notebooks/proyecto_cnn_aes.ipynb).  
> **Estado actual:** etapas 1–5 ejecutadas y verificadas en `develop`: evaluación test CNN (97,95 %), comparación con la MLP (94,51 %), Feature Maps y Grad-CAM. Los archivos de evidencia están subidos a `reports/` e `images/`. Falta revisión final del informe y preparación de la presentación; el benchmark homogéneo de tiempos es opcional.  
> **La información de más abajo corresponde a la EP1 y su MLP original.**

---

# Reconocimiento de Lenguaje de Señas (ASL) — Clasificación de las letras A, E y S

**Curso:** TLY1102 — Machine Learning Avanzado
**Dataset base:** Sign Language MNIST (Kaggle)
**Variante del grupo:** subconjunto de 3 clases (A, E, S), seleccionadas por ser las letras con formas de
mano más difíciles de distinguir entre sí.

---

## 1. Descripción del problema de negocio

El lenguaje de señas (LSA/ASL) es la principal vía de comunicación de personas sordas. Un sistema capaz de
reconocer automáticamente letras del alfabeto dactilológico permite una comunicación más directa entre
personas sordas y oyentes que no conocen el lenguaje de señas, sin depender de un intérprete presente en
todo momento.

No todas las letras son igual de fáciles de reconocer: existen configuraciones de mano muy similares entre
sí (puños cerrados, dedos parcialmente doblados) que generan alta confusión. Este proyecto se enfoca
deliberadamente en el caso más desafiante — las letras **A**, **E** y **S** — en vez de resolver el
problema completo de 24 clases, priorizando aportar valor donde el problema es más difícil.

## 2. Objetivos del proyecto

- Construir un pipeline reproducible de clasificación de imágenes con un Perceptrón Multicapa (MLP).
- Entrenar un modelo capaz de distinguir entre las letras A, E y S a partir de una imagen de 28×28 píxeles.
- Evaluar el modelo con métricas robustas para problemas multiclase (no solo accuracy).
- Identificar y explicar los errores del modelo, así como las limitaciones estructurales de un MLP para
  tareas de visión computacional.

## 3. Definición de KPIs

| KPI | Descripción | Meta orientativa |
|---|---|---|
| Accuracy (test) | % de imágenes de test clasificadas correctamente entre A/E/S | > 90% (baseline aleatorio = 33%) |
| F1-score macro (test) | Balance entre precision y recall, ponderando igual cada una de las 3 clases | > 0.85 |
| Recall por clase | Detección específica de cada letra (evita que el modelo "ignore" una clase difícil) | > 0.80 en cada clase |
| Tasa de confusión entre pares | % de errores concentrados en un par específico de letras (p. ej. A↔S) | Identificar y documentar, no es una meta numérica sino un hallazgo a explicar |

## 4. Descripción de las fuentes de datos

- **Fuente:** [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) (Kaggle).
- **Formato:** dos archivos CSV (`sign_mnist_train.csv`, `sign_mnist_test.csv`), cada fila representa una
  imagen: columna `label` (entero) + 784 columnas `pixel1`...`pixel784` (imagen 28×28 aplanada).
- **Características verificadas** (ver nota de discrepancia abajo):
  - Imágenes en **escala de grises** (1 canal), 28×28 píxeles.
  - **24 clases** (letras A–Y del alfabeto, excluyendo J y Z porque en ASL real requieren movimiento).
  - **27.455** imágenes de entrenamiento y **7.172** imágenes de test.
- **Subconjunto usado en este proyecto:** solo las filas con `label` correspondiente a A (0), E (4) y S (18).

> **Nota sobre discrepancia con la propuesta inicial:** la descripción preliminar del dataset (entregada
> antes de inspeccionar los archivos) indicaba imágenes RGB, 42 clases y ~20.000 imágenes. Al cargar los
> archivos reales se confirmó que el dataset es en escala de grises, con 24 clases y ~34.627 imágenes en
> total. El resto del proyecto se desarrolla con las características reales, verificadas directamente
> sobre los datos.

## 5. Preparación y análisis exploratorio de los datos (EDA)

Resumen (ver detalle y visualizaciones en el notebook, sección b):

- Se inspeccionó la distribución de clases del dataset completo para entender el contexto general antes
  de filtrar.
- Se visualizaron ejemplos individuales y múltiples muestras por clase de A, E y S, confirmando
  visualmente la similitud geométrica entre estas tres letras (configuraciones de puño cerrado).
- Se identificó variabilidad intra-clase por iluminación, ángulo de la mano y tono de piel, factor
  relevante para anticipar errores del modelo.

## 6. Preparación y transformación de datos para el modelamiento

1. Filtrado del dataset completo a solo las filas de A, E y S.
2. Remapeo de labels originales (0, 4, 18) a índices consecutivos (0, 1, 2) para la capa `softmax`.
3. Normalización de píxeles: división por 255 (rango 0–1).
4. Separación de un conjunto de **validación** (20% del train, estratificado por clase), manteniendo el
   **test** original intacto para la evaluación final.
5. Codificación one-hot de las etiquetas (`to_categorical`) para usar `categorical_crossentropy`.

## 7. Metodología utilizada (CRISP-DM)

Este proyecto sigue las fases de **CRISP-DM** (Cross Industry Standard Process for Data Mining):

1. **Comprensión del negocio**: definición del problema de comunicación que motiva el proyecto y de la
   variante específica (A/E/S) elegida por el grupo.
2. **Comprensión de los datos**: inspección de los archivos reales del dataset, corrección de la
   descripción inicial y EDA visual (distribución de clases, ejemplos, variabilidad intra-clase).
3. **Preparación de los datos**: filtrado, remapeo de labels, normalización, split train/val/test,
   codificación one-hot.
4. **Modelamiento**: diseño y entrenamiento de un MLP (arquitectura, función de pérdida, optimizador e
   hiperparámetros documentados y justificados en el notebook, sección d).
5. **Evaluación**: métricas de clasificación multiclase (accuracy, precision, recall, F1 macro, matriz de
   confusión) e interpretación en función del problema de negocio.
6. **Despliegue** (alcance del curso): entrega del modelo entrenado serializado (`models/mlp_asl_aes.keras`)
   y del notebook reproducible; no se implementa un despliegue productivo.

## 8. Consideraciones éticas

- El dataset no incluye metadatos demográficos explícitos, pero la variabilidad de tono de piel observada
  en el EDA es relevante: un modelo entrenado sin diversidad suficiente en los datos de origen podría
  desempeñarse peor en subgrupos subrepresentados. Esto debe tenerse en cuenta antes de cualquier uso real
  del modelo con población diversa.
- Este es un proyecto académico y el modelo **no está validado para uso en contextos reales de asistencia
  a personas sordas**; su alcance es exclusivamente educativo (comprensión del flujo de trabajo de ML
  supervisado).

## 9. Estructura del proyecto

```
asl-mlp-aes/
├── README.md                              <- este archivo
├── data/
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
├── notebooks/
│   └── proyecto_lenguaje_señas.ipynb       <- notebook reproducible (secciones a-h)
├── models/
│   └── mlp_asl_aes.keras                   <- se genera al ejecutar el notebook
└── images/                                 <- gráficos exportados al ejecutar el notebook
```

## 10. Cómo reproducir

1. Instalar dependencias: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`.
2. Abrir `notebooks/proyecto_lenguaje_señas.ipynb` (Jupyter o Google Colab).
3. Ejecutar todas las celdas en orden. Las rutas de datos son relativas (`../data/`), por lo que el
   notebook debe ejecutarse manteniendo la estructura de carpetas indicada arriba.
