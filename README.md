# Predicción de incumplimiento en préstamos Lending Club — Grupo 9

Proyecto de Machine Learning (UTEC). Se predice si un préstamo **aceptado a 36 meses** acabará en Charged Off/Default (1) o Fully Paid (0). No es un sistema de aprobación.

## Estado

Esta integración reúne la infraestructura de Valentino/Kris, el script y los resultados de árboles de Nicolás, y repara el módulo de árboles ausente. La entrega previa se conserva en proposal.md, notebooks/01_exploracion_inicial.ipynb y outputs/metrics_baseline.json. Sus métricas corresponden a otra cohorte: no comparar directamente con v4. Los resultados v4 son de validación; **test permanece reservado**. Leer reports/informe_final.md para resultados y pendientes.

## Estructura

| Ruta | Propósito |
|---|---|
| src/data.py, src/splits.py, src/features.py | Cohorte, particiones y lista blanca de variables |
| src/train.py, src/models_tree.py, src/models_boosting.py, src/evaluate.py | Modelos, pipelines y evaluación |
| scripts/ | Auditoría, preparación, entrenamiento y análisis de errores |
| tests/ | Pruebas sintéticas de contratos y modelos |
| notebooks/ | Exploración de entrega previa |
| reports/, outputs/ | Decisiones, informe y resultados |

## Reproducción desde cero en PowerShell

Desde la raíz del repositorio: descargar el [CSV de Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) y colocarlo en data/raw/accepted_2007_to_2018Q4.csv.gz. No subir el CSV ni las credenciales. Los scripts procesan todo el archivo por fragmentos.

~~~powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.audit_dataset
.\.venv\Scripts\python.exe -m scripts.audit_cohort_detail
.\.venv\Scripts\python.exe -m scripts.prepare_data --mode matured_36
.\.venv\Scripts\python.exe -m scripts.freeze_splits
.\.venv\Scripts\python.exe -m scripts.train_baseline_v4
.\.venv\Scripts\python.exe -m scripts.train_trees_nicolas --with-g2
.\.venv\Scripts\python.exe -m scripts.train_boosting_kris --n-iter 30 --with-gb
.\.venv\Scripts\python.exe -m scripts.feature_importance_boosting_kris
.\.venv\Scripts\python.exe -m scripts.analyze_errors
.\.venv\Scripts\python.exe -m scripts.analyze_errors --valid data/splits/lc_36m_v4/valid.parquet
.\.venv\Scripts\python.exe -m scripts.plot_validation
~~~

Entrenar todo puede tardar y requiere memoria/espacio suficientes. data/splits/lc_36m_v4/ contiene train/valid y contrato; nunca test.parquet. No se entrega un comando de test final porque falta fijar el modelo y el protocolo conjunto.

## Contrato y comparación

Train: 2007–junio 2014 (244 943); validación: julio–diciembre 2014 (90 615); test reservado: enero–junio 2015 (120 790). Solo préstamos de 36 meses con desenlace terminal bajo un corte 2018Q4 **supuesto**. La separación es por fecha de originación; el desenlace puede conocerse después, así que no es una simulación prospectiva estricta. Ver reports/contrato_dataset_v4.md.

G1: 18 atributos de originación; G2: tasa, cuota y grado de Lending Club como experimento aparte; G3: variables posteriores prohibidas por fuga. Se ajusta cada transformación dentro del Pipeline con train. Se comparan Dummy, logística, árbol, bosque y boosting mediante ROC-AUC/PR-AUC. La prevalencia de impago en validación es 14.13 %.

## Pendientes para una entrega final completa

Fijar modelo y umbral; analizar errores por segmentos y casos problemáticos; abrir una sola vez el test reservado con protocolo temporal explícito; incorporar resultados al informe y revisar la presentación. El CSV original no está en Git: esta integración valida código y reproduce análisis de predicciones guardadas, pero no reentrena con los datos originales.
