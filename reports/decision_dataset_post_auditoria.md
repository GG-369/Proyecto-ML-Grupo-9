# Grupo 9 — Evaluación de la auditoría exhaustiva (26-10-2026)

**Evidencia**: `outputs/audit_lending_club.json` facilitado por el equipo, generado contra `accepted_2007_to_2018Q4.csv.gz` (392,582,231 bytes). Ningún modelo ha sido reentrenado después de esta auditoría.

## Conteos comprobados

| Selección | Registros | Porcentaje del CSV | Nota |
|---|---:|---:|---|
| Archivo original | 2,260,701 | 100% | Incluye estados sin desenlace |
| Avance 1, `legacy_final` | 287,640 | 12.72% | Enero 2012–junio 2014, dos plazos |
| `matured_36_final` | 456,348 | 20.19% | Solo 36 meses, origen hasta junio 2015 |
| `matured_final` | 489,185 | 21.64% | 36 meses hasta junio 2015; 60 meses hasta junio 2013 |
| `all_final` | 1,345,350 | 59.51% | **No** permitido como cohorte entrenable sin estudio de censura |

El incremento de **58.65%** de `matured_36_final` y **70.07%** de `matured_final` respecto al Avance 1 compara tamaños de cohortes distintas: NO debe llamarse automáticamente "filas recuperadas". Para conocer los registros efectivamente añadidos hay que calcular los solapamientos.

Entre préstamos que cumplen el criterio temporal de maduración (`matured_date` = 491,935), los estados finales son 489,185 (99.44%). Para 36 meses la proporción es 456,348 / 458,787 = 99.47%. Esta es solo una validación de cobertura de estados finales; no demuestra ausencia de sesgo de selección ni confirma la fecha exacta de snapshot.

## Decisión provisional

**Primera opción a estudiar para la cohorte principal**: `matured_36_final` (456,348), ya que evita mezclar plazos con ventanas de originación asimétricas. Cambia el alcance: el modelo se interpretaría exclusivamente para préstamos aceptados a 36 meses. Si el objetivo del grupo exige ambos plazos, comparar con `matured_final` y documentar las diferencias de composición. No aprobar todavía la selección final: falta el conteo por año, estado y plazo.

## Obligatorio antes de reentrenar

1. Ejecutar `python -m scripts.audit_cohort_detail` y conservar `outputs/audit_cohort_detail.json`.
2. Comprobar la representación de incumplimientos en train/valid y en el holdout reservado, más el solapamiento con `legacy`.
3. Justificar la fecha de corte asumida `2018-12-31`: el nombre del archivo describe las originaciones, no confirma por sí solo una fecha exacta de extracción de estados.
4. Congelar definición de población objetivo, fechas y nombres de features en un contrato común del equipo.
5. Conservar un test cerrado: **no usar su rendimiento para escoger modelos ni ajustar hiperparámetros**.
6. En la defensa, describir la partición por `issue_d` como validación retrospectiva por fecha de originación. No prometer una simulación prospectiva estricta sin verificar cuándo se conocieron las etiquetas de entrenamiento.

## Regla de go/no-go

Si el informe detallado muestra cambios importantes en `bad_rate_among_final`, composición temporal insuficiente o una caída sustancial del número de positivos en la ventana de validación, mantener `legacy_final` como control mientras se investiga. Nunca entrenar con `all_final` solo porque tiene más registros.

## Trabajos independientes

- **Valentino:** cohorte y contratos, validación reproducible, datos y baseline oficial; documentar claramente el cambio de población.
- **Nicolás:** revisar consistencia de atributos, faltantes y preparación de datos; modelos de árboles si el equipo mantiene el reparto acordado.
- **Kris:** gradient boosting y comparación contra baseline con la MISMA cohorte y las MISMAS fechas.
- **Rubí:** segmentación de errores, sesgos, auditabilidad y análisis de variables; elaborar el código antes del resultado final.

**Sin resultados inventados**: los resultados de las regresiones logísticas del Avance 1 no pueden reutilizarse como métricas de la nueva cohorte.
