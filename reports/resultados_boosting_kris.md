# Resultados de boosting — Kris (rama `feat/boosting-kris`)

Contrato de datos: `lc_36m_2007_2015h1_v4`. Todos los números de este informe son de la **validación congelada** (90,615 préstamos, emisión julio–diciembre 2014). El test reservado (120,790) **no se ha usado**.

## 1. Resumen

- `HistGradientBoostingClassifier` supera a la regresión logística de Valentino en ROC-AUC y PR-AUC con ambos grupos de variables, pero **la mejora es pequeña**: +0.0117 ROC-AUC con G1 y +0.0059 con G1+G2.
- En un bootstrap pareado sobre las filas de validación, el intervalo del 95% de la diferencia queda por encima de cero en las cuatro comparaciones (sección 6). Ese intervalo solo recoge la variabilidad del conjunto de validación, no la de la semilla ni la del período.
- El ranking de ambos modelos es muy parecido: el 63% (G1) y el 71% (G1+G2) de los préstamos del 10% más riesgoso coinciden.
- Con umbral 0.5 el boosting casi no marca préstamos riesgosos (F1 = 0). No es un fallo: es consecuencia de no usar `class_weight='balanced'` en las configuraciones ganadoras (sección 8).
- El modelo sigue sin estar listo para decisiones crediticias reales.

## 2. Qué se hizo

- **Modelo:** `HistGradientBoostingClassifier` de scikit-learn, dentro de un `Pipeline` (`src/models_boosting.py`).
- **Variables:** G1 (comparación obligatoria, 18 variables) y G1+G2 (experimento separado, suma `int_rate`, `installment` y `grade`), tomadas de `src.features.select_feature_columns`. Ninguna variable de desempeño posterior.
- **Preprocesamiento (ajustado solo con train):** numéricas convertidas a `float64` sin escalar, con `NaN` manejado de forma nativa por el modelo; categóricas con codificación ordinal declaradas como categóricas.
- **Datos:** `train.parquet` (244,943) y `valid.parquet` (90,615).
- **Código:** `src/models_boosting.py`, `scripts/train_boosting_kris.py`.

## 3. Búsqueda de hiperparámetros

**Protocolo.** Búsqueda aleatoria de 30 configuraciones (semilla 42), las mismas 30 para G1 y G1+G2. Cada una se ajusta con préstamos de train emitidos antes de 2013-07-01 y se puntúa en el resto de train (julio 2013 a junio 2014), la **subvalidación temporal**. El número de iteraciones sale de la mejor ROC-AUC de la subvalidación (`staged_predict_proba`, cada 10 iteraciones, tope 500). La configuración ganadora se reentrena con todo train y se evalúa **una vez** en la validación congelada, que no interviene en la elección.

**Espacio de búsqueda:** `learning_rate` entre 0.02 y 0.2 (escala logarítmica), `max_leaf_nodes` ∈ {8, 15, 31, 63}, `max_depth` ∈ {sin límite, 4, 6, 8}, `min_samples_leaf` ∈ {20, 50, 100, 200, 500}, `l2_regularization` ∈ {0, 0.1, 1, 10}, `class_weight` ∈ {ninguno, balanced}. Elegí rangos estándar para boosting, que controlan el tamaño de los árboles, la velocidad de aprendizaje y la regularización. Las mejores configuraciones quedaron en los bordes, así que ampliar el espacio es un paso pendiente.
**Configuraciones elegidas**

| Parámetro | G1 | G1+G2 |
|---|---:|---:|
| learning_rate | 0.03278 | 0.02287 |
| max_leaf_nodes | 15 | 8 |
| max_depth | 4 | 8 |
| min_samples_leaf | 500 | 50 |
| l2_regularization | 10.0 | 10.0 |
| class_weight | ninguno | ninguno |
| max_iter (mejor en subvalidación) | 490 | 500 |
| ROC-AUC en subvalidación | 0.6509 | 0.6667 |

**Qué muestra la búsqueda**

- **Las 30 configuraciones rindieron casi igual.** En la subvalidación, la ROC-AUC va de 0.6453 a 0.6509 con G1 (desviación estándar 0.0014) y de 0.6615 a 0.6667 con G1+G2 (0.0013). Las diferencias entre las mejores son del orden de 0.0003, así que **la elección de la ganadora no es concluyente**: otra de las configuraciones del grupo de cabeza habría dado un resultado prácticamente igual.
- **`class_weight`:** sin pesos, la ROC-AUC media fue algo mayor (G1: 0.6497 con 16 configuraciones contra 0.6483 con 14 con `balanced`; G1+G2: 0.6655 contra 0.6643). La diferencia es chica, pero consistente en ambos grupos.
- **Tope de iteraciones:** la ganadora de G1 usó 490 de 500 y la de G1+G2 llegó al tope de 500. El tope pudo limitar el resultado. No se probó un tope mayor.
- El árbol de G1+G2 tiene `max_depth=8` con solo 8 hojas, así que el límite de profundidad no restringe nada en la práctica.
- Tiempo de la búsqueda (suma de los `seconds` de cada configuración): unos 133 s para G1 y 152 s para G1+G2. El reentrenamiento final tardó 3.0 s y 3.1 s. Máquina no registrada.

Detalle completo: `search_boosting_kris_g1.csv` y `search_boosting_kris_g1_g2.csv`.

## 4. Resultados en validación (90,615 préstamos; 14.13% incumplen)

| Variables | Modelo | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| — | Dummy | 0.5000 | 0.1413 |
| G1 | Regresión logística | 0.6548 | 0.2231 |
| G1 | **HGB** | **0.6666** | **0.2329** |
| G1+G2 | Regresión logística | 0.6794 | 0.2450 |
| G1+G2 | **HGB** | **0.6853** | **0.2512** |

Diferencia HGB menos regresión logística: G1 +0.0117 (ROC-AUC) y +0.0098 (PR-AUC); G1+G2 +0.0059 y +0.0062. En ambos modelos, agregar G2 mejora más que cambiar de modelo.

No se deben comparar con el Avance 1 (0.685 de ROC-AUC): cambiaron la población, las fechas y el plazo.

## 5. Comparación a igual tasa de marcado

Como el umbral 0.5 no es comparable entre modelos con y sin `class_weight`, se compara marcando como riesgoso el mismo porcentaje de préstamos con mayor puntaje. Recall = fracción de los incumplimientos que quedan dentro de los marcados.

| % marcado | Variables | Regresión logística (precisión / recall) | HGB (precisión / recall) |
|---:|---|---:|---:|
| 10% | G1 | 0.263 / 0.186 | 0.280 / 0.198 |
| 10% | G1+G2 | 0.296 / 0.210 | 0.300 / 0.212 |
| 20% | G1 | 0.241 / 0.341 | 0.249 / 0.352 |
| 20% | G1+G2 | 0.258 / 0.366 | 0.263 / 0.372 |
| 30% | G1 | 0.223 / 0.474 | 0.230 / 0.488 |
| 30% | G1+G2 | 0.237 / 0.503 | 0.238 / 0.506 |
| 50% | G1 | 0.195 / 0.690 | 0.200 / 0.707 |
| 50% | G1+G2 | 0.204 / 0.722 | 0.206 / 0.729 |

El 10% con mayor puntaje concentra entre el 18.6% y el 21.2% de los incumplimientos (lift entre 1.86 y 2.12 sobre el azar). HGB queda por encima en todas las filas, pero con G1+G2 las diferencias son de uno o dos puntos porcentuales o menos.

## 6. Robustez de la diferencia (bootstrap pareado)

300 remuestreos con reemplazo de las 90,615 filas de validación (semilla 42), calculando en cada uno la diferencia HGB menos regresión logística sobre las mismas filas.

| Comparación | Diferencia media | IC 95% |
|---|---:|---:|
| ROC-AUC, G1 | +0.0117 | [0.0092, 0.0140] |
| PR-AUC, G1 | +0.0098 | [0.0066, 0.0131] |
| ROC-AUC, G1+G2 | +0.0058 | [0.0041, 0.0076] |
| PR-AUC, G1+G2 | +0.0061 | [0.0029, 0.0092] |

**Cómo leerlo:** si se volviera a muestrear un conjunto de validación parecido, la ventaja del boosting se mantendría. **No dice** que se mantenga con otra semilla, otra ventana temporal o el test, y las filas de un mismo período no son independientes entre sí.

## 7. Solapamiento entre modelos

- Top 10% (9,062 préstamos): coinciden 5,707 en G1 (63.0%) y 6,468 en G1+G2 (71.4%).
- Correlación de Spearman entre las puntuaciones de ambos modelos: 0.894 (G1) y 0.944 (G1+G2).

Los dos modelos ordenan los préstamos de forma bastante parecida, lo que concuerda con una mejora pequeña. Por qué ocurre no está probado; una posibilidad es que haya poca estructura no lineal por aprovechar con estas variables, pero no se comprobó.

## 8. Resultados con umbral 0.5 (provisional)

| Variables | Modelo | Precisión | Recall | F1 | Predichos riesgosos |
|---|---|---:|---:|---:|---:|
| G1 | Regresión logística | 0.195 | 0.688 | 0.304 | 49.8% |
| G1 | HGB | 0.000 | 0.000 | 0.000 | 0.002% |
| G1+G2 | Regresión logística | 0.212 | 0.666 | 0.322 | 44.3% |
| G1+G2 | HGB | 0.000 | 0.000 | 0.000 | 0% |

Matrices de confusión de HGB (filas = real 0/1, columnas = predicho 0/1): G1 `[[77813, 2], [12800, 0]]`; G1+G2 `[[77815, 0], [12800, 0]]`.

Las puntuaciones del HGB sin pesos de clase se concentran cerca de la prevalencia (14%): media 0.132 en G1 y 0.126 en G1+G2, percentil 99 en 0.32 y máximo 0.516 y 0.440, por lo que casi ninguna llega a 0.5. **Esos F1 y recall no son comparables** con los de la regresión logística, que usa `class_weight='balanced'`. La accuracy (0.8587) coincide con la del Dummy y no informa nada. Para comparar de forma justa se usan ROC-AUC, PR-AUC y la tabla de la sección 5.

## 9. Limitaciones

1. Una sola semilla (42) y una sola partición temporal; sin intervalos que incluyan la variabilidad de entrenamiento.
2. La elección de hiperparámetros no es concluyente: las configuraciones de cabeza empatan en la subvalidación (sección 3). El tope de 500 iteraciones pudo limitar el resultado.
3. Las puntuaciones no están calibradas como probabilidades reales de impago; el umbral 0.5 es provisional y no hay estimación de costes de error.
4. Validación retrospectiva con *snapshot* 2018Q4 supuesto: los préstamos aún activos quedaron excluidos, con riesgo de sesgo de selección y censura. La población son solo préstamos aceptados a 36 meses.
5. Los resultados con G1+G2 incorporan la evaluación crediticia previa de Lending Club (`grade`, `int_rate`), por lo que no son equivalentes a los de G1.
6. No se analizaron errores por segmento ni posibles sesgos; eso corresponde a Rubí.

## 10. Reproducibilidad

```bash
python -m scripts.train_boosting_kris --n-iter 30
```

Semilla 42; Python 3.14.8, scikit-learn 1.9.0, pandas 3.0.4, numpy 2.5.3. Salidas en `outputs/`: `metrics_boosting_kris_v4.json`, `predictions_valid_boosting_kris.csv`, `search_boosting_kris_g1.csv`, `search_boosting_kris_g1_g2.csv`. Las comparaciones de las secciones 5 a 7 se calcularon aparte con `predictions_valid_v4.csv` y `predictions_valid_boosting_kris.csv`, unidos por `id` (relación 1:1, 90,615 filas, mismo `target`); ese cálculo no está en el repositorio.

## 11. Pendiente / no ejecutado

- `GradientBoosting` clásico (`--with-gb`): **no se ejecutó**.
- Corridas con otras semillas y búsqueda con un tope de iteraciones mayor a 500: **no se hicieron**.
- Importancia de variables: **no calculada**.
- Evaluación sobre el test reservado: **no realizada**; se hará una sola vez al final, con el modelo ya elegido por el equipo.
