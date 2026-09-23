# Predicción de incumplimiento en préstamos P2P (Lending Club)

Proyecto final del curso CS3061 — Machine Learning, UTEC.
**Grupo 9:** Nicolás, Kris, Rubí, Valentino.

---

## 1. Título del proyecto

Predicción de incumplimiento en préstamos P2P (Lending Club) con validación temporal y control de leakage.

## 2. Integrantes

Grupo 9 — Nicolás, Kris, Rubí, Valentino.

## 3. Dataset elegido

- **Fuente:** [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) en Kaggle (usuario `wordsforthewise`), que republica los datos abiertos originalmente liberados por la propia plataforma Lending Club.
- **Archivo usado:** `accepted_2007_to_2018Q4.csv` — préstamos aceptados entre 2007 y 2018, con más de 2 millones de filas y ~150 columnas en su versión completa.
- **Tamaño tras el filtro que usamos:** 287,640 filas y 154 columnas (ver sección 8).
- **Por qué es un dataset complejo:** tiene más de 50 mil filas y más de 50 variables, mezcla datos numéricos, categóricos y temporales, tiene valores faltantes reales (no simulados) y requiere decisiones explícitas de qué variables son válidas para evitar leakage — justo los problemas de "mundo real" que pide el curso.
- **Decisión de muestreo:** se usó solo el rango de préstamos emitidos entre **enero 2012 y junio 2014**, y solo aquellos con desenlace final (`Fully Paid`, `Charged Off` o `Default`). Ver sección 6 para el porqué.
- **Licencia:** CC0-1.0 (Public Domain).

## 4. Pregunta predictiva

Con la información disponible al momento en que Lending Club aprueba un préstamo, ¿qué probabilidad hay de que ese préstamo termine en incumplimiento (`Charged Off` o `Default`) en vez de pagarse completo (`Fully Paid`)?

## 5. Variable objetivo

`target`: variable binaria construida a partir de `loan_status`.

- `target = 1` si `loan_status` es `Charged Off` o `Default`.
- `target = 0` si `loan_status` es `Fully Paid`.

Se excluyen del dataset los préstamos en estados que no son finales (`Current`, `Late`, `In Grace Period`, y las variantes `Does not meet the credit policy...`), porque todavía no tienen un desenlace conocido; incluirlos obligaría a inventar una etiqueta.

## 6. Unidad de predicción

Un préstamo individual, evaluado en el momento en que Lending Club lo aprueba (antes de que empiece a pagarse).

## 7. Variables disponibles antes de la predicción

Clasificamos todas las columnas en 4 grupos (detalle completo en `reports/tabla_variables.csv`):

| Grupo | Descripción | Ejemplos | ¿Se usa? |
|---|---|---|---|
| G0 | Identificadores, texto libre o proxies de alta cardinalidad | `id`, `member_id`, `url`, `desc`, `title`, `emp_title`, `zip_code` | No |
| G1 | Disponibles al momento de aprobar el préstamo | `loan_amnt`, `term`, `emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `purpose`, `addr_state`, `dti`, `delinq_2yrs`, `earliest_cr_line`, `fico_range_low/high`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc` | Sí, base del baseline |
| G2 | Asignadas por el propio Lending Club al aprobar | `grade`, `sub_grade`, `int_rate`, `installment` | Se prueba por separado (con y sin), ver sección 12 |
| G3 | Solo existen después de que el préstamo ya avanzó o terminó | `total_pymnt`, `total_rec_prncp`, `recoveries`, `last_pymnt_d`, `out_prncp`, columnas `hardship_*` y `settlement_*` | No — son leakage |

## 8. Riesgos de leakage

Las variables del grupo G3 describen eventos posteriores a la aprobación del préstamo (pagos ya recibidos, recuperaciones tras impago, fechas de últimos pagos). Usarlas equivaldría a que el modelo "vea la respuesta" antes de predecir.

**Evidencia:** entrenamos la misma regresión logística dos veces, una solo con variables G1 y otra agregando algunas de G3. El ROC-AUC pasó de 0.685 a 0.999, un salto que confirma que esas columnas filtran directamente el resultado en vez de aportar señal legítima de riesgo.

## 9. Métrica principal y métrica secundaria

- **Principal:** ROC-AUC. Mide qué tan bien el modelo ordena los préstamos por riesgo, sin depender de un umbral de decisión fijo.
- **Secundarias:** PR-AUC (más sensible al desbalance de clases), F1 y matriz de confusión con umbral 0.5.
- La accuracy se reporta solo como referencia: con ~83.6% de préstamos pagados, un modelo que siempre predice "pagará" ya tiene ~82.4% de accuracy (en validación) sin aportar nada útil.

## 10. Plan de validación

Partición **temporal**, no aleatoria: entrenamos con préstamos emitidos en 2012-2013 y validamos con los emitidos en el primer semestre de 2014. Esto simula el uso real del modelo, que siempre predice sobre préstamos futuros a partir de patrones del pasado; una partición aleatoria mezclaría épocas y sobreestimaría el desempeño real.

## 11. Modelo baseline

Dos referencias:

1. `DummyClassifier` (siempre predice la clase mayoritaria) — referencia mínima.
2. Regresión logística dentro de un `Pipeline` de scikit-learn (imputación + escalado/codificación + modelo), en dos variantes: solo con variables G1, y con G1 + G2.

**Resultados** (`outputs/metrics_baseline.json`):

| Modelo | ROC-AUC | PR-AUC | F1 | Accuracy |
|---|---|---|---|---|
| Dummy | 0.500 | — | — | 0.824 |
| Regresión logística (G1) | 0.685 | 0.309 | 0.377 | 0.608 |
| Regresión logística (G1 + G2) | 0.699 | 0.320 | 0.389 | 0.628 |

**Lectura del resultado:** El baseline con variables G1 alcanza un ROC-AUC de 0.685, superando claramente al modelo aleatorio (0.500). Esto significa que, tomando al azar un préstamo que no se pagó y uno que sí, el modelo asigna mayor riesgo al que no se pagó el 68.5% de las veces. Al agregar las variables propietarias `grade` e `int_rate` (G2), el AUC sube a 0.699; este ligero incremento tiene sentido porque dichas variables representan un puntaje de riesgo que Lending Club ya calculó internamente, facilitando la tarea predictiva del algoritmo al proveerle información preprocesada por la plataforma.

## 12. Exploración de datos

- **Filas y columnas tras el filtro:** 287,640 filas y 154 columnas.
- **Tipos de datos:** 118 numéricas (float64/int64), 35 categóricas (object) y 1 de fecha (datetime64).
- **Valores faltantes:** 32 columnas tienen 100% de valores faltantes (ej. `member_id` eliminada por privacidad, y `sec_app_*` que solo aplican a préstamos conjuntos, los cuales casi no existieron en este periodo).
- **Distribución del target:** 83.6% de la clase mayoritaria (pagó) frente a un 16.4% de impago, confirmando un fuerte desbalance.
- **Hallazgos principales de los gráficos:**
  - **Tendencia temporal:** La tasa de impago varía por mes de emisión, justificando la partición temporal.
  - **Grade y Purpose:** Existe una relación progresiva entre el grado del préstamo y el riesgo (mayor riesgo en `G` que en `A`). El propósito del préstamo discrimina claramente categorías más riesgosas.
  - **Ingreso y FICO:** A mayor puntaje FICO, el riesgo disminuye de forma drástica y consistente. Los prestatarios morosos muestran ingresos ligeramente inferiores.
- **Outliers detectados:** Se encontraron 24 registros con `annual_inc` ≤ 0 o mayores a 1 millón, y 534 casos de `revol_util` > 100%. No se borraron de inmediato, pero requerirán manejo de outliers en la etapa de feature engineering.

## 13. Riesgos técnicos

- El archivo original pesa más de 1 GB; se procesa en trozos (`chunksize`) para no saturar la memoria.
- El desbalance de clases (~84/16) puede sesgar métricas simples como accuracy; por eso se prioriza ROC-AUC y PR-AUC.
- La ventana de tiempo elegida (2012-2014) es un balance entre tener suficientes préstamos ya resueltos y no perder representatividad.
- Tiempo de trabajo comprimido a un día: se prioriza dejar cada decisión documentada y reproducible sobre explorar variantes adicionales.

## 14. Riesgos éticos, sesgos y limitaciones

1. **Sesgo de selección:** El dataset solo contiene préstamos *aceptados*; no hay forma de evaluar a la población rechazada.
2. **Proxies de variables sensibles:** El ingreso y la geografía (`addr_state`) pueden actuar como proxy indirecto de raza o género (no incluidos).
3. **Datos autorreportados:** Variables como `annual_inc` y `emp_length` son declaradas por el usuario y no siempre están verificadas.
4. **Ventana temporal acotada:** Entrenado en la economía de EE.UU. 2012-2014; no generaliza a crisis económicas futuras ni a otros países.
5. **Dependencia del propio modelo de Lending Club:** Al incluir G2 (`grade`, `int_rate`), el modelo aprende a imitar la evaluación de riesgo de la plataforma en lugar de predecir desde cero.
6. **Desbalance de clases:** Un clasificador ingenuo alcanza alta accuracy, requiriendo pesos de clase balanceados y métricas estrictas como ROC-AUC.

## 15. Plan de trabajo para las semanas restantes

| Semanas | Objetivo |
|---|---|
| 2-3 | Limpieza definitiva y feature engineering; pasar el código de los notebooks a scripts (`src/data.py`, `src/features.py`) |
| 4-6 | Entrenar al menos 3 familias de modelos (regresión logística regularizada, árboles/random forest, gradient boosting) con validación temporal y búsqueda de hiperparámetros |
| 9-10 | Tratamiento del desbalance (pesos de clase, ajuste de umbral, muestreo), análisis de errores por segmento, interpretabilidad |
| 11-12 | Evaluación final sobre el conjunto de test bloqueado (préstamos de 2014-H2, no tocado hasta esta etapa), figuras finales |
| 13-14 | Informe final y presentación |
| 15-16 | Margen y ensayo de la presentación |

## Estructura del repositorio

```text
Proyecto-ML-Grupo-9/
├── README.md
├── proposal.md
├── requirements.txt
├── notebooks/
│   └── 01_exploracion_inicial.ipynb
├── reports/
│   └── tabla_variables.csv
└── outputs/
    └── metrics_baseline.json
```

No se incluyen archivos de datos (`.csv`/`.parquet`): son demasiado grandes para el repositorio y se regeneran siguiendo las instrucciones de abajo.

## Instrucciones de reproducción

1. Crear una cuenta en Kaggle y generar un token de API (`Cuenta → Create New API Token`), que descarga un archivo `kaggle.json`.
2. Abrir `notebooks/01_exploracion_inicial.ipynb` en Google Colab.
3. Subir `kaggle.json` cuando la primera celda lo solicite; esa celda descarga automáticamente el dataset `wordsforthewise/lending-club` desde Kaggle.
4. Ejecutar el notebook de principio a fin (`Entorno de ejecución → Ejecutar todas`). El notebook filtra el dataset por la ventana de tiempo definida (sección 6 de este README), construye el `target`, corre la exploración y el baseline, y guarda las métricas.
5. Semilla aleatoria fijada en `42` en todos los pasos con componente aleatorio, para que los resultados sean reproducibles.

## Referencias

- Dataset: Lending Club Loan Data, vía Kaggle (`wordsforthewise/lending-club`).
- Bibliografía del curso: Bishop, *Pattern Recognition and Machine Learning* (2007); Müller & Guido, *Introduction to Machine Learning with Python* (2016).