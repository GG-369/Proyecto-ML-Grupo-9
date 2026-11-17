# Contratos de trabajo en paralelo — Grupo 9

**Regla:** nadie modifica el notebook original, el informe del compañero ni `src/config.py` sin una PR de coordinación. La rama `main` actual es la referencia del Avance 1.

## Preparación común (Valentino; este avance)

- `src/config.py`: objetivo binario, estatus válidos, cohortes comparables, semilla 42.
- `src/data.py`: auditoría exhaustiva de 2M+ filas y exportación con allowlist.
- `src/features.py`: derivaciones deterministas G1, preprocesamiento fit dentro del Pipeline.
- `src/train.py`, `src/evaluate.py`: baseline y métricas de VALIDACIÓN, nunca test.
- `scripts/`: CLI para VS Code/Colab; `tests/`: tests de contratos.
- **Por ejecutar:** el dataset real no está subido a este repositorio.

## Trabajos independientes, cuando la cohorte y split estén acordados

| Integrante | Rama sugerida | Archivos exclusivos | Entregable / prueba de aceptación |
|---|---|---|---|
| **Nicolás** | `feat/modelos-arboles-nicolas` | `notebooks/03_modelos_arboles.ipynb`, `src/models_tree.py`, `outputs/metrics_tree_*.json` | RandomForest / árboles; métricas en **la misma validación**, tiempo y semillas |
| **Kris** | `feat/boosting-kris` | `notebooks/03_modelos_boosting.ipynb`, `src/models_boosting.py`, `outputs/metrics_boost_*.json` | Gradient boosting / HistGradientBoosting; comparación con dummy y logística sobre el mismo split |
| **Rubí** | `feat/auditoria-rubi` | `notebooks/04_analisis_errores.ipynb`, `reports/analisis_sesgos.md`, `reports/figures/` | Auditoría de leakage y disponibilidad, interpretación y errores segmentados usando predicciones de validación |
| **Valentino** | `feat/infra-valentino` | `src/config.py`, `src/data.py`, `src/features.py`, `src/train.py`, `src/evaluate.py`, `scripts/`, `tests/`, `README.md` | Auditoría completa, contrato de cohortes, baseline reproducible, mantener particiones y test bloqueado |

**Rubí no necesita esperar a que termine un modelo final:** puede trabajar con la regresión logística ya disponible y datos sintéticos de tests estructurales para montar las funciones, pero el análisis final de errores debe usar predicciones reales de validación. Nicolás y Kris pueden entrenar en paralelo una vez fijado el parquet/split común.

## Convención común para resultados

Cada experimento deberá documentar `dataset_mode`, periodo train/valid,
features G1 o G1+G2, semilla, modelo, hiperparámetros, número de filas, ROC-AUC,
PR-AUC, F1 y matriz de confusión. No se publican nuevas cifras hasta ejecutar
contra datos reales. Nadie abre el test para elegir modelos.

## Orden recomendado de merge

1. `feat/infra-valentino` después de tests y auditoría real; 2. ramas Nicolás/Kris/Rubí, sin editar archivos comunes; 3. revisión técnica conjunta y congelación de test; 4. evaluación final, informe y presentación.
