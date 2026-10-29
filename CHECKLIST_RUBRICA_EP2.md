# Lista de verificación contra rúbrica — EP2 CNN (2026)

Esta lista no es una calificación oficial. Contrasta la pauta de **Presentación y defensa técnica del proyecto** con las evidencias que están disponibles en `develop`. Sección de referencia principal: [informe técnico EP2](INFORME_TECNICO_EP2.md).

| Indicador de evaluación | Peso | Evidencias comprobadas | Estado / acción |
|---|---:|---|---|
| IE1. Continuidad del problema y variante propia | 15 % | Mismo dataset EP1, subconjunto A/E/S y justificación, propósito CNN; semilla 42 y partición explicada | **Cubierto** |
| IE2. Preparación de datos para CNN | 15 % | CSV reales, EDA, escala de grises 28×28, normalización /255, train/val/test y separación de test | **Cubierto** |
| IE3. Diseño e implementación de la CNN | 25 % | Dos bloques Conv2D/Pooling, Dense, ReLU, Adam, crossentropy; dos configuraciones, notebook ejecutado, checkpoints .keras | **Cubierto** |
| IE4. Comparación con el modelo MLP | 15 % | Mismo test, accuracy/precision/recall/F1, parámetros, tablas y gráficos, argumentos y limitaciones | **Parcial:** falta benchmark homogéneo del **tiempo MLP vs CNN** y su interpretación en el informe/presentación |
| IE5. Evaluación del desempeño y análisis de errores | 20 % | Curvas train/val, accuracy/precision/recall/F1, matriz, casos correctos e incorrectos, dificultad S→E | **Cubierto** |
| IE6. Comunicación técnica y defensa | 10 % | Presentación en PowerPoint, notas y enlaces a resultados | **Por ensayar:** defensa oral individual de hasta 10 minutos y respuestas técnicas |

## Acciones antes de dar el trabajo por cerrado

1. **Tiempo de entrenamiento MLP/CNN (IE4):** en la sección **25** del notebook `notebooks/proyecto_cnn_aes.ipynb`, cambiar temporalmente `EJECUTAR_BENCHMARK_TIEMPOS = False` por `True`. Ejecutar esa celda en **Google Colab** con los CSV, TensorFlow y las rutas `RAIZ_ETAPA4`, `DIR_REPORTES_E4` ya inicializadas por la etapa 4. El script entrena tres modelos nuevos con mismo entorno, particiones, batch, optimizador y callbacks y exporta `reports/benchmark_tiempos_mlp_cnn.csv`. Esta medición es independiente de los checkpoints ya evaluados; no modifica la selección ni el test original. Subir el CSV, interpretar el tiempo y actualizar el informe/diapositiva de comparación. Registrar el dispositivo usado (CPU/GPU).
2. **Reproducibilidad:** conservar `data/`, `models/`, `reports/`, `images/`, el informe EP2 y notebook ejecutado. Evitar sobreescribir checkpoints elegidos con el benchmark.
3. **Defensa:** ensayar en menos de 10 minutos. Poder justificar selección A/E/S, convolución vs MLP, pooling, criterios de selección por validación, errores S→E, interpretaciones Feature Maps/Grad-CAM y riesgo de fotografías aumentadas similares.
4. **Publicación:** una vez aprobadas las comprobaciones, abrir un Pull Request de **`BenjaDaDuoc/asl-cnn-aes:develop` hacia `BenjaDaDuoc/asl-cnn-aes:main`** (no hacia el fork de origen `KrisCea/asl-mlp-aes`). No fusionar antes de la revisión final.

## Resultados que sí están verificados

- **MLP:** 1.016/1.075 aciertos = **94,51 %**, F1 macro **93,65 %**.
- **CNN seleccionada:** 1.053/1.075 = **97,95 %**, F1 macro **97,72 %**, 219.779 parámetros.
- **Diferencia:** +**3,44 puntos porcentuales** de accuracy, **37** aciertos adicionales.
- **Dificultad mayor:** clase S; 21 casos S→E y recall **91,46 %**.
- **Validación:** ambas CNN alcanzaron 100 %, pero esto no garantiza generalización a datos externos.
- **Tiempo medido hasta ahora:** CNN base **87,44 s** y CNN con más filtros **129,38 s**, comparables entre sí en esa sesión. **No existe aún tiempo comparable de MLP**.

**Advertencia de interpretación:** la rúbrica evalúa la calidad de la explicación personal, no solamente que los archivos existan. Los estados «cubierto» son observaciones sobre evidencia técnica, no garantía de puntaje.
