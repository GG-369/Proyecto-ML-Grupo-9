# Modelos de árboles — Nicolás

## Alcance y protocolo

Se comparan `DecisionTreeClassifier` y `RandomForestClassifier` con el contrato
`lc_36m_2007_2015h1_v4`. G1 es el experimento obligatorio; G1+G2 solo se ejecuta
como comparación separada. El código carga únicamente `train.parquet` y
`valid.parquet`, valida sus fechas, IDs y tamaños contra `contract.json`, y
rechaza la presencia de `test.parquet`.

La selección de hiperparámetros ocurre en una subvalidación temporal interna de
train: se ajusta con fechas anteriores al 1 de julio de 2013 y se escoge la
configuración con mayor ROC-AUC en las fechas posteriores de train. Después,
cada modelo elegido se vuelve a ajustar con todo train y se evalúa una sola vez
en la validación congelada. La semilla es 42. Imputación y codificación se
ajustan dentro del pipeline, nunca antes de separar los datos.

## Búsqueda de hiperparámetros

- **Árbol:** cuatro configuraciones; profundidad máxima de 6, 12 o sin límite,
  tamaño mínimo de hoja de 100 o 500 préstamos y peso de clases balanceado.
- **Random Forest:** cuatro configuraciones; 200 árboles, profundidad de 12 o
  sin límite, tamaño mínimo de hoja de 100 o 300, `max_features=sqrt` y pesos
  balanceados por muestra bootstrap.
- La selección por profundidad y tamaño de hoja controla el sobreajuste. El
  bosque promedia muchos árboles; `random_state=42` hace reproducible su
  aleatoriedad y `n_jobs=-1` usa los procesadores disponibles.
- La comparación principal usa G1. G1+G2 agrega tasa de interés, cuota y grado
  crediticio como experimento distinto; sus resultados no se deben mezclar con
  G1 al elegir el mejor modelo.

## Resultados en validación

La auditoría del CSV aceptado encontró 2,260,701 filas y 456,348 préstamos
finalizados a 36 meses (62,092 incumplimientos). La validación congelada
contiene 90,615 préstamos, con una prevalencia de incumplimiento de 14.13%.
Se ejecutaron G1 (comparación obligatoria) y G1+G2 (experimento separado).

### G1 — Variables disponibles al aprobar (comparación principal)

| Modelo | Features | ROC-AUC | PR-AUC | Precision | Recall | F1 | Captura top 10% | Ajuste final (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (referencia) | — | 0.5000 | 0.1413 | 0.0000 | 0.0000 | 0.0000 | 0.00% | — |
| Regresión logística | G1 | 0.6548 | 0.2231 | — | — | 0.3042 | — | — |
| Decision Tree | G1 | 0.6323 | 0.2084 | 0.1873 | 0.6696 | 0.2927 | 17.74% | 9.1 |
| Random Forest | G1 | 0.6560 | 0.2238 | 0.2054 | 0.6097 | 0.3073 | 18.92% | 20.7 |

### G1+G2 — Agrega `int_rate`, `installment`, `grade` (experimento separado)

| Modelo | Features | ROC-AUC | PR-AUC | Precision | Recall | F1 | Captura top 10% | Ajuste final (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | G1+G2 | 0.6658 | 0.2290 | 0.2052 | 0.6639 | 0.3135 | 19.32% | 3.8 |
| Random Forest | G1+G2 | 0.6831 | 0.2478 | 0.2203 | 0.6312 | 0.3266 | 21.08% | 22.8 |

El umbral para precision, recall y F1 es 0.5. ROC-AUC es la métrica principal;
PR-AUC complementa la evaluación ante la menor frecuencia de incumplimientos.
La captura top 10% mide qué proporción de todos los incumplimientos aparece en
el décimo de préstamos con mayor score.

**Hiperparámetros seleccionados en subvalidación temporal (issue_d < 2013-07-01):**

- **Decision Tree G1:** `max_depth=12`, `min_samples_leaf=500`, `class_weight=balanced` (ROC-AUC interna 0.6198)
- **Random Forest G1:** `n_estimators=200`, `max_depth=None`, `min_samples_leaf=100`, `max_features=sqrt`, `class_weight=balanced_subsample` (ROC-AUC interna 0.6488)
- **Decision Tree G1+G2:** `max_depth=6`, `min_samples_leaf=100`, `class_weight=balanced` (ROC-AUC interna 0.6514)
- **Random Forest G1+G2:** `n_estimators=200`, `max_depth=None`, `min_samples_leaf=100`, `max_features=sqrt`, `class_weight=balanced_subsample` (ROC-AUC interna 0.6674)

La semilla es 42 en todos los casos.

**Análisis G1:** El Random Forest queda ligeramente por encima de la regresión logística G1 en
esta validación (+0.0012 ROC-AUC y +0.0007 PR-AUC), mientras que el árbol queda
por debajo. La diferencia del bosque es pequeña, así que no demuestra por sí
sola una mejora relevante. Con umbral 0.5, el árbol detecta más incumplimientos
(recall 66.96%) pero marca como riesgosos más préstamos que sí se pagaron
(precision 18.73%). Para Random Forest, la matriz de confusión, con filas de
clase real `[pago, incumplimiento]` y columnas de predicción `[pago,
incumplimiento]`, es `[[47629, 30186], [4996, 7804]]`; para el árbol es
`[[40629, 37186], [4229, 8571]]`.

**Análisis G1+G2:** Al agregar las variables propietarias de Lending Club (`grade`, `int_rate`, `installment`), ambos modelos mejoran respecto a G1. Random Forest G1+G2 alcanza **ROC-AUC 0.6831** (+0.0271 vs RF G1, +0.0283 vs logística G1) y **captura top-10% 21.08%**. La mejora al añadir G2 es mayor que la ganancia por cambiar de algoritmo (logística → RF). Nota: G1+G2 incorpora la evaluación crediticia previa de Lending Club, por lo que no es comparable directamente con G1 para decidir el mejor modelo base.

## Cómo reproducir

Desde la raíz del proyecto, después de generar los splits locales:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_trees_nicolas
```

Para ejecutar además el experimento separado G1+G2:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_trees_nicolas --with-g2
```

El proceso genera `outputs/metrics_tree_nicolas_v4.json`,
`outputs/predictions_valid_tree_nicolas.csv` y archivos
`outputs/search_*_nicolas_*.csv`. El JSON registra hiperparámetros ganadores,
métrica interna, ROC-AUC, PR-AUC, precision, recall, F1, matriz de confusión,
captura top 10%, tiempos, versiones y tamaños de partición. La corrida reportada
usó Python 3.14.7, scikit-learn 1.9.1, pandas 3.1.0 y NumPy 2.5.4.

## Limitaciones

La evaluación es retrospectiva y depende del corte asumido para las etiquetas;
no representa una simulación perfecta de qué etiquetas estaban disponibles en
cada fecha. La validación sirve para comparar modelos, no para tomar decisiones
crediticias reales. El conjunto de test sigue reservado y no se usa para escoger
configuraciones.