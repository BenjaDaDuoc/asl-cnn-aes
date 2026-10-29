# EP2 — CNN de reconocimiento de letras A, E y S

**Asignatura:** Técnicas Avanzadas de Machine Learning I (TLY1102)  
**Problema:** clasificación de imágenes de signos estáticos del alfabeto dactilológico estadounidense (ASL).  
**Variante del equipo:** letras **A, E y S**, igual que en la Evaluación Parcial 1 (EP1).

> **Estado:** etapa 1 implementada — preparación del dataset y análisis exploratorio (EDA). Todavía **no se ha entrenado ni evaluado una CNN**. Este documento acompaña el trabajo en `develop`; el README original de la MLP se conserva en esta etapa.

## Archivo principal

- `notebooks/proyecto_cnn_aes.ipynb`: notebook documentado de la segunda evaluación.
- `notebooks/proyecto_lenguaje_señas.ipynb`: MLP original de la EP1; sirve como línea base.
- `data/sign_mnist_train.csv`, `data/sign_mnist_test.csv`: dataset original ya incluido en el repositorio.
- `models/mlp_asl_aes.keras`: modelo anterior; **no se sobrescribe**.
- `requirements.txt`: dependencias de esta etapa.

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

## Próximas etapas

1. Diseñar y justificar una arquitectura CNN (Conv2D, pooling y capas densas), junto con al menos una alternativa para comparar hiperparámetros.
2. Entrenar con semilla reproducible, EarlyStopping y registro de curvas de accuracy/loss; investigar posibles casi duplicados.
3. Evaluar en el test original con accuracy, precision, recall, F1, matriz de confusión y ejemplos correctos/incorrectos.
4. Comparar MLP y CNN en métricas, parámetros, tiempo y generalización. Visualizar Feature Maps y mapas de activación de cada clase.
5. Documentar resultados reales, limitaciones, decisiones éticas y conclusiones, preparar la defensa de hasta diez minutos.

## Alcance responsable

No es un sistema para uso real con personas sordas, ni reconoce secuencias dinámicas de señas. En particular, clasificar signos estáticos de ASL no equivale a traducir Lengua de Señas Chilena (LSCh). La revisión de errores por iluminación, encuadre y diversidad de manos es parte de la evaluación.
