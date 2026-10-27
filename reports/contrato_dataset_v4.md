# Contrato de datos v4 — Préstamos Lending Club a 36 meses

**Estado:** candidato adoptado como contrato operativo para el Avance 2/3; no es una evaluación prospectiva perfecta ni despliegue real.

## Justificación verificable

Auditoría original: 2,260,701 observaciones. Cohorte legacy: 287,640 (36 y 60 meses). Cohorte 36m madura y con estado final: 456,348 (394,256 pagados, 62,092 incumplidos); incidencia 13.61%. La cohorte 36m nueva no es una ampliación pura del conjunto original: hay 215,847 casos compartidos, 240,501 casos nuevos respecto a legacy y se excluyen 71,793 legacy de 60 meses.

Criterio de entrada: plazo original de 36 meses, originación hasta junio de 2015, desenlace entre Fully Paid / Charged Off / Default, con supuesto de corte del archivo a fin de 2018 y margen de seis meses. Esta política **no prueba** seguimiento perfecto ni certifica que el estado fuera conocido al empezar el período de validación.

## Particiones congeladas (fecha de emisión issue_d)

| Parte | Ventana | n | Intervención permitida |
|---|---|---:|---|
| Entrenamiento | 2007-01 a 2014-06 | 244,943 | `fit` de preprocesamiento/modelo |
| Validación | 2014-07 a 2014-12 | 90,615 | Comparación, selección e hiperparámetros |
| Test reservado | 2015-01 a 2015-06 | 120,790 | Solamente evaluación final conjunta |

Entrenamiento incluye 29,096 observaciones de 2007–2011 (información más antigua con distinta completitud potencial). **Ablación predefinida**: comparar en la misma validación el modelo entrenado sobre todo 2007–2014H1 frente a entrenar únicamente desde 2012, sin abrir test. No elegir features mediante observación del test.

Tasas de incumplimiento de la auditoría (solo descriptivas, no métricas de modelo): train ≈12.67%, valid ≈14.13%, test ≈15.12%. Hay drift temporal en la proporción de etiquetas.

## Riesgos pendientes / alcance de la afirmación

1. **Fecha de disponibilidad del desenlace:** un préstamo 36m emitido en 2014 puede finalizar años después. Entrenar en él y simular una decisión de julio 2014 sería leakage temporal de la etiqueta. Los splits de emisión son válidos como **benchmark retrospectivo con separación temporal de origen**, no como *backtest point-in-time* auténtico.
2. **Censura y sesgo de selección:** al excluir préstamos activos y seleccionar vencidos respecto al snapshot supuesto, queda riesgo residual por finalización anticipada, retrasos y variación por época.
3. **Características al otorgamiento:** permitir solo G1 para baseline; G2 como comparador contextual. Validar timestamps originales con diccionario del dataset antes de sumar más features.
4. **Alcance del objetivo:** predicción de incumplimiento entre créditos *aprobados de 36 meses*, no decisión sobre solicitudes rechazadas ni préstamos de 60 meses.
5. **Complejidad vs volumen:** más registros no garantizan menor error; comprobar AUC/PR-AUC y drift solo en validación.

## Contrato técnico para el equipo

- `scripts.prepare_data --mode matured_36` materializa Parquet en `data/processed/matured_36/`.
- `scripts.freeze_splits` exige auditoría detallada y snapshot previsto; exporta **train.parquet y valid.parquet** a `data/splits/lc_36m_v4/`, **nunca test.parquet**.
- `scripts.train_baseline_v4` compara Dummy, logística G1 y logística G1+G2 y genera `outputs/metrics_validacion_v4.json` y `outputs/predictions_valid_v4.csv` para Rubí.
- Nicolás y Kris leen la misma carpeta `data/splits/lc_36m_v4/` y solo validan sus modelos contra `valid.parquet`. Rubí estudia métricas/error en validación y revisa variables; no requiere esperar los otros modelos.
- Ningún modelo acepta `test` como argumento. Una evaluación de test conjunta se definirá después de fijar decisiones de modelado, con resultados archivados y semilla fija.

## Cambios respecto al Avance 1

Se conservan `notebooks/01_exploracion_inicial.ipynb`, `proposal.md` y `outputs/metrics_baseline.json` como evidencia histórica; la propuesta deberá actualizarse al finalizar la revisión para describir el alcance de 36 meses y nuevas fechas de validación. La cifra ROC-AUC 0.685 corresponde únicamente al Avance 1; **no compararla directamente con modelos del nuevo contrato**, pues las poblaciones y ventanas cambiaron.
