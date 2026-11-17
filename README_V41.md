# Parche 4.1 — Lectura correcta de métricas y ablación temporal

## Hallazgo verificado

En `metrics_validacion_v4.json`, ROC-AUC, PR-AUC, F1 y accuracy coinciden con las 90,615 filas de `predictions_valid_v4.csv`. **`positive_rate: 0.0` era incorrecto** por `sum(y_true)` sobre etiquetas NumPy `int8`, acumulación susceptible de overflow, y por un nombre de métrica ambiguo.

Ahora `src/evaluate.py` calcula `target_prevalence` (clase real) y `predicted_positive_rate` (predicciones >= threshold), manteniendo `positive_rate` como alias retrocompatible de `target_prevalence`. Incluye precision y recall. No cambia el entrenamiento ni las etiquetas ni las puntuaciones del modelo.

## Ejecutar tras extraer el parche sobre el repositorio

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.train_baseline_v4
.\.venv\Scripts\python.exe -m scripts.compare_training_windows_v41
```

El segundo comando actualiza el JSON v4 (con métricas arregladas). El tercer comando produce `outputs/comparison_training_windows_v41.json`. Compara **exactamente la misma validación** (julio–diciembre de 2014) con el train completo 2007–junio de 2014 (244,943 filas) y train reducido 2012–junio de 2014 (215,847 filas). Reentrena G1 y G1+G2 con idéntico preprocesador. **No lee test**.

## Cómo interpretar

- `delta_roc_auc_full_minus_recent > 0`: el modelo entrenado con todos los años ordena mejor el riesgo, en esa validación.
- `delta_pr_auc_full_minus_recent > 0`: mejor desempeño de precision-recall, en esa validación.
- Si las diferencias son pequeñas o contrapuestas, evitar afirmaciones causales: la comparación cambia a la vez los años y el tamaño del entrenamiento. Ante empate práctico, priorizar la cohorte más representativa y documentar la decisión.
- **No seleccionar mirando test enero–junio 2015.** Mantener sellado.

## Para Rubí

El CSV de predicciones v4 solo contiene `id,issue_d,target,score_*`. Para analizar sesgos por `addr_state`, ingresos o FICO, cruzar con **`data/splits/lc_36m_v4/valid.parquet`** por `id` (validar cardinalidad 1:1 y ausencia de nulos). Trabajar solo en validación; no subir datos sensibles/grandes a GitHub.

## Límites que permanecen

Los splits por `issue_d` son retrospectivos: no demuestran que se conocieran las etiquetas al inicio de cada ventana. El snapshot de 2018Q4 sigue siendo supuesto; la selección por estado final puede producir sesgo por censura/selección. G2 es solo un comparador con decisiones de Lending Club. Mantener comparaciones con el baseline del Avance 1 separadas porque usan diferentes poblaciones.
