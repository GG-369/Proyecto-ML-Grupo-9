# Traspaso de tareas — v4.2 (Grupo 9)

## Fuente común

`data/splits/lc_36m_v4/train.parquet` (244,943) y `valid.parquet` (90,615), obtenidos localmente con los scripts del repositorio; `contract.json` debe indicar `lc_36m_2007_2015h1_v4`. Estos archivos están ignorados por Git: **los compañeros deben regenerarlos con el CSV de Kaggle y el archivo de auditoría** (o recibirlos por un canal privado autorizado). `test.parquet` **no debe existir** en esa carpeta. Para la auditoría: `python -m scripts.audit_cohort_detail`, luego `prepare_data --mode matured_36`, `freeze_splits`, `train_baseline_v4`.

## Ramas y archivos exclusivos

| Persona | Rama | Archivos que edita | Evidencia de aceptación |
|---|---|---|---|
| Valentino | `feat/infra-valentino` | `src/`, `scripts/`, `tests/` comunes; `reports/decision_train_v42.md` | Contrato de datos + documentación y verificación reproducible, baseline G1 y G1+G2 |
| Nicolás | `feat/arboles-nicolas` | `src/models_tree.py`, `scripts/train_trees_nicolas.py`, `reports/resultados_arboles_nicolas.md` | DecisionTree y RandomForest, ROC-AUC/PR-AUC sobre la validación fija, hiperparámetros y semilla documentados |
| Kris | `feat/boosting-kris` | `src/models_boosting.py`, `scripts/train_boosting_kris.py`, `reports/resultados_boosting_kris.md` | HistGradientBoosting/Gradient Boosting, justificación de búsqueda, métricas comparables |
| Rubí | `feat/auditoria-rubi` | `src/error_analysis.py`, `scripts/error_analysis_rubi.py`, `reports/errores_sesgos_rubi.md` | FP/FN, matriz, segmentos, proxies y limitaciones; unir validación por ID 1:1, nunca por orden de filas |

**Regla de independencia:** nadie modifica el mismo archivo que otro, y cada uno usa `train.parquet`, `valid.parquet` y `contract.json` sin redefinir cohortes ni tocar test. Si Rubí necesita scores de árboles/boosting, puede comenzar con `outputs/predictions_valid_v4.csv` del baseline y luego incorporar salidas nuevas cuando existan; **no debe quedar bloqueada**.

## Contrato de comparación

- `target=1` es incumplimiento; `target=0` es pago completo; `id` identifica préstamos.
- Métrica principal ROC-AUC; secundarias PR-AUC, precision, recall, F1, matriz de confusión, detección en top decile.
- Comparar siempre en la validación congelada; no remezclar train/valid. Pipelines con imputación y encoders entrenados en train, no antes del split. Registrar semilla 42, biblioteca/versiones, cantidad de filas y tiempo.
- Reportar G1 como comparación justa obligatoria. G1+G2 solo como experimento separado, no competir modelos con grupos de variables distintos sin señalarlo.
- Hiperparámetros: preferir subvalidación temporal **dentro de train** o búsqueda bien documentada; no elegir el mejor modelo basándose en test.
- Para Rubí: hacer `merge(..., on='id', validate='one_to_one')` con `valid.parquet`. Dado que `predictions_valid_v4.csv` no tiene FICO ni ingresos, no inventar variables segmentadas.
- Entregar código ejecutable y métricas de validación reproducibles; resultados parciales o no ejecutados deben declararse como tales.

## Condición de integración (PR)

Cada integrante crea su rama desde el último commit común en `main`, sube únicamente sus archivos propios y abre PR; no hacer cambios cruzados no acordados a `features.py`, `train.py`, `splits.py` o `requirements.txt`. Valentino revisa que `python -m pytest -q` pase, que no se suban datos masivos ni credenciales, y que nadie abra el test.

## Trabajo reservado para etapa final

Unificar modelos bajo la misma cohorte y features, justificar selección usando validación, análisis interpretativo y de errores, escribir informe, y evaluar sobre test **una sola vez** tras congelar todas las decisiones de modelado. Esta última evaluación aún no se ha realizado.
