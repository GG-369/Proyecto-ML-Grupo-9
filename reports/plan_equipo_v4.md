# Reparto de trabajo sin conflictos (contrato v4)

Todos los integrantes trabajan con `data/splits/lc_36m_v4/{train,valid}.parquet`; **nadie utiliza test**. La carpeta se genera localmente; nunca se sube la data a GitHub.

| Persona | Rama | Archivos propios | Salida |
|---|---|---|---|
| Valentino | `feat/infra-valentino` | `src/splits.py`, `scripts/freeze_splits.py`, `scripts/train_baseline_v4.py`, `tests/test_split_contract_v4.py` | Contrato, pruebas, Dummy/logística G1 y G1+G2; validación sin test |
| Nicolás | `feat/arboles-nicolas` | `src/models_tree.py`, `notebooks/03_arboles_nicolas.ipynb` | DecisionTree/RandomForest y métricas de validación |
| Kris | `feat/boosting-kris` | `src/models_boosting.py`, `notebooks/03_boosting_kris.ipynb` | Boosting, búsqueda con partición train/valid, métricas comparables |
| Rubí | `feat/auditoria-rubi` | `src/error_analysis.py`, `notebooks/04_errores_rubi.ipynb`, `reports/error_sesgo_v4.md` | Matrices de confusión, análisis por segmentos, documentación de leakage con `predictions_valid_v4.csv` |

**Convenciones:** `target=1` es incumplimiento; métrica principal ROC-AUC, secundaria PR-AUC, F1, matriz de confusión y precisión/recall. Fijar `random_state=42`; imputación/encoding dentro de Pipelines ajustados solo en train. Versionar hiperparámetros y el `contract_id=lc_36m_2007_2015h1_v4`. Documentar diferencias de features y la ablation histórica.

**Criterio de aceptación de branches:** código que consume train y valid congelados, pruebas propias básicas, resultados de validación reproducibles y ninguna lectura de filas de test. PRs separadas; Valentino coordina la integración final. Si falta el dataset local, entregar código sin afirmar nuevas métricas.
