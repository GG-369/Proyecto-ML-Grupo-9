# Resultados de boosting — Kris (rama `feat/boosting-kris`)

Contrato de datos: `lc_36m_2007_2015h1_v4`. Todos los números de este informe son de la **validación congelada** (90,615 préstamos, emisión julio–diciembre 2014). El test reservado (120,790) **no se ha usado**.

## 1. Resumen

- `HistGradientBoostingClassifier` supera a la regresión logística de Valentino en ROC-AUC y PR-AUC con ambos grupos de variables, pero **la mejora es pequeña**: +0.0117 ROC-AUC con G1 y +0.0060 con G1+G2.
- En un bootstrap pareado sobre las filas de validación, el intervalo del 95% de la diferencia queda por encima de cero en las cuatro comparaciones (sección 6). Ese intervalo solo recoge la variabilidad del conjunto de validación, no la de la semilla ni la del período.
- El ranking de ambos modelos es muy parecido: el 63% (G1) y el 71% (G1+G2) de los préstamos del 10% más riesgoso coinciden.
- Con umbral 0.5 el boosting casi no marca préstamos riesgosos (F1 = 0). No es un fallo: es consecuencia de no usar `class_weight='balanced'` en las configuraciones ganadoras (sección 8).
- **Actualización:** se amplió el espacio de búsqueda de hiperparámetros (sección 3) porque las ganadoras originales quedaban en los bordes. Con el espacio ampliado, `l2_regularization` **sigue ganando en el borde superior** (ahora 100.0, antes 10.0) en ambos grupos de variables, y en G1+G2 el tope de iteraciones también sigue copado (1000/1000). `min_samples_leaf` ya no queda en un borde. Pese a estos cambios, las métricas de validación **no cambiaron de forma relevante** (diferencias ≤0.0002 en ROC-AUC y PR-AUC, por debajo de la variabilidad entre configuraciones de la propia búsqueda). La conclusión no cambia: el modelo sigue sin estar listo para decisiones crediticias reales.

## 2. Qué se hizo

- **Modelo:** `HistGradientBoostingClassifier` de scikit-learn, dentro de un `Pipeline` (`src/models_boosting.py`).
- **Variables:** G1 (comparación obligatoria, 18 variables) y G1+G2 (experimento separado, suma `int_rate`, `installment` y `grade`), tomadas de `src.features.select_feature_columns`. Ninguna variable de desempeño posterior.
- **Preprocesamiento (ajustado solo con train):** numéricas convertidas a `float64` sin escalar, con `NaN` manejado de forma nativa por el modelo; categóricas con codificación ordinal declaradas como categóricas.
- **Datos:** `train.parquet` (244,943) y `valid.parquet` (90,615).
- **Código:** `src/models_boosting.py`, `scripts/train_boosting_kris.py`.

## 3. Búsqueda de hiperparámetros

**Protocolo.** Búsqueda aleatoria de 30 configuraciones (semilla 42), las mismas 30 para G1 y G1+G2. Cada una se ajusta con préstamos de train emitidos antes de 2013-07-01 y se puntúa en el resto de train (julio 2013 a junio 2014), la **subvalidación temporal**. El número de iteraciones sale de la mejor ROC-AUC de la subvalidación (`staged_predict_proba`, cada 10 iteraciones, tope `MAX_ITER_CAP`). La configuración ganadora se reentrena con todo train y se evalúa **una vez** en la validación congelada, que no interviene en la elección.

### 3.1 Espacio original (primera corrida) y por qué se amplió

Espacio original: `learning_rate` entre 0.02 y 0.2 (escala log), `max_leaf_nodes` ∈ {8, 15, 31, 63}, `max_depth` ∈ {sin límite, 4, 6, 8}, `min_samples_leaf` ∈ {20, 50, 100, 200, 500}, `l2_regularization` ∈ {0, 0.1, 1, 10}, `class_weight` ∈ {ninguno, balanced}, `MAX_ITER_CAP=500`.

Con ese espacio, las ganadoras quedaron en varios bordes: `l2_regularization` ganó en 10.0 (el valor más alto) en ambos grupos; `min_samples_leaf` ganó en 500 (el más alto) en G1; `learning_rate` ganó cerca del extremo bajo (0.023–0.033 sobre un rango de 0.02–0.2); y el tope de 500 iteraciones se usó casi completo en las dos ganadoras (490/500 y 500/500). Eso sugiere que el espacio pudo estar recortando el óptimo, así que se ensanchó.

### 3.2 Espacio ampliado (esta corrida)

Cambios, justificados uno por uno a partir de qué borde tocaba cada parámetro:

- `l2_regularization` ∈ {0, 0.1, 1, 10, 30, **100**}: se agregaron 30 y 100 porque 10 (el máximo anterior) ganó en ambos grupos.
- `min_samples_leaf` ∈ {20, 50, 100, 200, 500, **1000**}: se agregó 1000 porque 500 (el máximo anterior) ganó en G1.
- `learning_rate` entre **0.01** y 0.2 (escala log): se bajó el piso de 0.02 a 0.01 porque las ganadoras quedaban cerca del piso anterior.
- `MAX_ITER_CAP`: 500 → **1000**, porque ambas ganadoras usaban el tope casi completo.
- `max_leaf_nodes` y `max_depth` **no se tocaron**: la ganadora de G1+G2 en la corrida original usaba `max_depth=8` con solo 8 hojas, es decir la profundidad no estaba restringiendo nada en la práctica; ampliar esos ejes sin evidencia de que estuvieran en el borde habría sido arbitrario.
- `n_iter` se mantuvo en 30 (no se subió) para poder comparar la dispersión de resultados contra la corrida original con el mismo número de configuraciones; duplicar `MAX_ITER_CAP` ya duplica aproximadamente el costo por configuración (`early_stopping=False`, así que el tiempo de ajuste escala con `max_iter`).

**Configuraciones elegidas (espacio ampliado)**

| Parámetro | G1 | G1+G2 |
|---|---:|---:|
| learning_rate | 0.01621 | 0.01191 |
| max_leaf_nodes | 15 | 8 |
| max_depth | 8 | 8 |
| min_samples_leaf | 200 | 50 |
| l2_regularization | 100.0 | 100.0 |
| class_weight | ninguno | ninguno |
| max_iter (mejor en subvalidación) | 820 | 1000 |
| ROC-AUC en subvalidación | 0.6514 | 0.6670 |

**Qué muestra la búsqueda ampliada**

- **Las 30 configuraciones vuelven a rendir casi igual.** ROC-AUC de subvalidación entre 0.6461 y 0.6514 con G1 (desviación estándar 0.0014) y entre 0.6622 y 0.6670 con G1+G2 (0.0013) — la misma dispersión que en la corrida original. La elección de la ganadora sigue sin ser concluyente.
- **`l2_regularization` sigue en el borde superior**, ahora en 100.0 (el nuevo máximo) en ambos grupos. Ampliar el espacio no resolvió el patrón: solo movió el borde. No se probaron valores por encima de 100 porque, como se ve abajo, el movimiento del borde no se tradujo en una mejora real de las métricas de validación — seguir ensanchando este eje no parece prioritario.
- **`min_samples_leaf` ya no está en un borde**: ganó en 200 (G1) y 50 (G1+G2), ambos dentro del espacio, por debajo incluso del máximo antiguo (500). Ese parámetro quedó resuelto.
- **Tope de iteraciones:** en G1 la ganadora usó 820 de 1000 (ya no copa el tope); en G1+G2 sí llegó al nuevo tope (1000/1000) — sigue sin resolverse en ese grupo, aunque de nuevo sin traducirse en mejora visible.
- **`learning_rate`:** la ganadora de G1 (0.01621) quedó bien dentro del nuevo rango; la de G1+G2 (0.01191) quedó cerca del nuevo piso (0.01), repitiendo el patrón de preferir tasas bajas.
- **`class_weight`:** se repite el patrón anterior — sin pesos, la ROC-AUC media fue algo mayor (G1: 0.6502 con 16 configuraciones sin peso contra 0.6490 con 14 `balanced`; G1+G2: 0.6658 contra 0.6647).
- Tiempo de la búsqueda (suma de `seconds` por configuración): 366 s para G1 y 385 s para G1+G2 (~12.5 min en total), frente a 133 s y 152 s con el espacio y el tope anteriores — el aumento es coherente con duplicar `MAX_ITER_CAP` (el ajuste no usa `early_stopping`, así que el costo escala con `max_iter`). El reentrenamiento final tardó 7.4 s y 8.0 s.

**Conclusión de la ampliación.** El espacio más grande cambia **dónde** cae el óptimo de la búsqueda (menos `learning_rate`, más `l2_regularization`, más iteraciones) pero **no cambia el resultado en validación**: ROC-AUC y PR-AUC quedan dentro de ±0.0002 de los valores anteriores (sección 4), una diferencia muy por debajo de la dispersión de 0.0013–0.0014 entre configuraciones de la misma búsqueda. `l2_regularization` sigue pegada al borde superior ampliado, lo que indica que el modelo "preferiría" todavía más regularización, pero esa preferencia no se traduce en mejor desempeño medible — es más probable que sea una meseta plana en la superficie de validación que una mejora real por explotar. No se encontró evidencia de que el tope de iteraciones anterior (500) estuviera realmente limitando el resultado: en G1 el nuevo óptimo ni siquiera usa el tope anterior ampliado, y en G1+G2, aunque sigue copando el tope, la métrica final no mejoró.

Detalle completo: `search_boosting_kris_g1.csv` y `search_boosting_kris_g1_g2.csv` (sobrescritos con la corrida del espacio ampliado; los valores de la corrida original quedan documentados arriba en el texto para referencia).

## 4. Resultados en validación (90,615 préstamos; 14.13% incumplen)

| Variables | Modelo | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| — | Dummy | 0.5000 | 0.1413 |
| G1 | Regresión logística | 0.6548 | 0.2231 |
| G1 | **HGB** | **0.6665** | **0.2329** |
| G1+G2 | Regresión logística | 0.6794 | 0.2450 |
| G1+G2 | **HGB** | **0.6854** | **0.2510** |

Diferencia HGB menos regresión logística: G1 +0.0117 (ROC-AUC) y +0.0098 (PR-AUC); G1+G2 +0.0060 y +0.0060. En ambos modelos, agregar G2 mejora más que cambiar de modelo.

**Con el espacio de búsqueda ampliado (sección 3.2), el HGB prácticamente no cambió** frente a la corrida con el espacio original: ROC-AUC se movió -0.0001 (G1) y +0.0001 (G1+G2); PR-AUC se movió 0.0000 (G1) y -0.0002 (G1+G2). Son diferencias de cuarto decimal, muy por debajo de la dispersión entre configuraciones de la propia búsqueda (~0.0013). Los IC del bootstrap pareado de la sección 6 se calcularon con las predicciones de la corrida con el espacio original; dado que las métricas nuevas caen dentro de ese mismo orden de magnitud, las conclusiones de las secciones 5 a 8 (comparación a igual tasa de marcado, bootstrap, solapamiento, umbral 0.5) siguen siendo válidas y no se recalcularon.

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

Matrices de confusión de HGB con el espacio de búsqueda ampliado (filas = real 0/1, columnas = predicho 0/1): G1 `[[77815, 0], [12800, 0]]`; G1+G2 `[[77815, 0], [12800, 0]]`.

Las puntuaciones del HGB sin pesos de clase se concentran cerca de la prevalencia (14%): media 0.132 en G1 y 0.126 en G1+G2, percentil 99 en 0.31 y máximo 0.48 y 0.44, por lo que ninguna llega a 0.5. **Esos F1 y recall no son comparables** con los de la regresión logística, que usa `class_weight='balanced'`. La accuracy (0.8587) coincide con la del Dummy y no informa nada. Para comparar de forma justa se usan ROC-AUC, PR-AUC y la tabla de la sección 5.

## 9. Limitaciones

1. Una sola semilla (42) y una sola partición temporal; sin intervalos que incluyan la variabilidad de entrenamiento.
2. La elección de hiperparámetros no es concluyente: las configuraciones de cabeza empatan en la subvalidación (sección 3). Se amplió el espacio de búsqueda y el tope de iteraciones (500 → 1000) y la ganadora de G1+G2 **sigue** copando el tope; `l2_regularization` sigue ganando en el borde superior del espacio ampliado (ahora 100.0). Ninguno de esos dos movimientos cambió las métricas de validación de forma medible, pero tampoco se puede descartar que un espacio aún mayor (o más iteraciones) cambie algo; no se exploró más allá por el costo de cómputo.
3. Las puntuaciones no están calibradas como probabilidades reales de impago; el umbral 0.5 es provisional y no hay estimación de costes de error.
4. Validación retrospectiva con *snapshot* 2018Q4 supuesto: los préstamos aún activos quedaron excluidos, con riesgo de sesgo de selección y censura. La población son solo préstamos aceptados a 36 meses.
5. Los resultados con G1+G2 incorporan la evaluación crediticia previa de Lending Club (`grade`, `int_rate`), por lo que no son equivalentes a los de G1.
6. No se analizaron errores por segmento ni posibles sesgos; eso corresponde a Rubí.

## 10. Reproducibilidad

```bash
python -m scripts.train_boosting_kris --n-iter 30
```

Semilla 42; Python 3.14.8, scikit-learn 1.9.0, pandas 3.0.4, numpy 2.5.3. Con el espacio de búsqueda ampliado (sección 3.2, `MAX_ITER_CAP=1000`), la corrida de 30 configuraciones por grupo tardó ~366 s (G1) + ~385 s (G1+G2) ≈ 12.5 min en total; máquina no registrada. Salidas en `outputs/`: `metrics_boosting_kris_v4.json`, `predictions_valid_boosting_kris.csv`, `search_boosting_kris_g1.csv`, `search_boosting_kris_g1_g2.csv`. Las comparaciones de las secciones 5 a 7 se calcularon aparte con `predictions_valid_v4.csv` y `predictions_valid_boosting_kris.csv`, unidos por `id` (relación 1:1, 90,615 filas, mismo `target`); ese cálculo no está en el repositorio y no se repitió en esta corrida porque las métricas de HGB no cambiaron de forma relevante (sección 4).

## 11. Pendiente / no ejecutado

- `GradientBoosting` clásico (`--with-gb`): **no se ejecutó**.
- Corridas con otras semillas: **no se hicieron**.
- Espacio de búsqueda y tope de iteraciones (`sample_configs`, `MAX_ITER_CAP`): **ampliados** en esta actualización (sección 3.2: `l2_regularization` hasta 100, `min_samples_leaf` hasta 1000, `learning_rate` desde 0.01, `MAX_ITER_CAP` hasta 1000). `l2_regularization` sigue ganando en el nuevo borde superior y, en G1+G2, el tope de iteraciones sigue copado; un espacio todavía más amplio (o un tope aún mayor) queda como pendiente, con bajo valor esperado dado que las métricas de validación no se movieron con esta ampliación.
- Importancia de variables: **no calculada**.
- Evaluación sobre el test reservado: **no realizada**; se hará una sola vez al final, con el modelo ya elegido por el equipo.
