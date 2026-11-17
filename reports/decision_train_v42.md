# Decisión de cohorte y entrenamiento — v4.2 (26/10/2026)

## Decisión que regirá para la comparación de modelos

- **Unidad de predicción:** préstamo aceptado de Lending Club con `term = 36` meses.
- **Target:** `1` si `loan_status` es `Charged Off` o `Default`, `0` si es `Fully Paid`. Otros estados sin etiqueta, no imputar su desenlace.
- **Cohorte:** `matured_36_final`, 456,348 registros; corte conservador por plazo con *snapshot* 2018Q4 **supuesto, no demostrado**.
- **Train principal:** 244,943 préstamos, emisión anterior a 2014-07-01 (2007–2014H1).
- **Validación común:** 90,615 préstamos, emisión desde 2014-07-01 hasta 2014-12-31.
- **Test reservado:** 120,790 préstamos, emisión desde 2015-01-01 hasta 2015-06-30; no exportar, leer ni utilizar para seleccionar modelos o umbrales.
- **Variables:** G1 (baseline oficial). G1+G2 (experimento secundario: incorpora información de calificación de Lending Club), reportado por separado. Ninguna característica G3, de desempeño posterior, entra al modelo.
- **Métrica principal:** ROC-AUC; secundarias PR-AUC, precision, recall, F1, matriz de confusión, métricas por segmento. Todas sobre la misma validación.
- **Parámetros de reproducción:** `random_state=42`, imputación, escalado y codificación dentro de pipelines entrenados solo con train.

## Evidencia de sensibilidad temporal (comparación v4.1)

La comparación reentrena modelos sobre dos ventanas y usa los **mismos** 90,615 registros de validación:

| Variante | Datos entrenamiento | G1 ROC-AUC | G1 PR-AUC | G1+G2 ROC-AUC | G1+G2 PR-AUC |
|---|---:|---:|---:|---:|---:|
| Completa 2007–2014H1 | 244,943 | 0.6548377 | 0.2230884 | 0.6793582 | 0.2449693 |
| Reciente 2012–2014H1 | 215,847 | 0.6559025 | 0.2237737 | 0.6808857 | 0.2471358 |

La ventana reciente vence por una diferencia **muy pequeña** de ROC-AUC (0.00106 en G1; 0.00153 en G1+G2). A umbral 0.5, con G1+G2: reciente genera 1,683 falsos positivos menos, pero 271 falsos negativos más (incumplimientos omitidos). No hay intervalos de incertidumbre ni estimación de costes de errores. **Conservar la ventana completa** para maximizar observaciones y evitar cambiar el contrato del equipo por una mejora marginal aún no robustecida. Conservar la ventana reciente como **análisis de sensibilidad**, no como baseline oficial.

Esta comparación observacional cambia época y tamaño de la muestra a la vez; **no demuestra causalidad** ni prueba que los años antiguos sean beneficiosos o perjudiciales. Fuente: `reports/comparison_training_windows_v41.json`, copia reproducible de la salida de `scripts.compare_training_windows_v41`.

## Limitaciones no resueltas (declararlas en la exposición)

1. Validación retrospectiva por fecha de emisión **no equivale** a una simulación de producción: los resultados de préstamos entrenados no necesariamente se conocían al iniciar la validación.
2. El momento exacto del *snapshot* de 2018Q4 no se verificó; los préstamos sin estado final no reciben etiqueta y hay riesgo de sesgo de selección/censura.
3. Los préstamos rechazados no forman parte de la población: no extrapolar la herramienta a toda persona solicitante.
4. Las puntuaciones de regresión logística con `class_weight=balanced` **no están calibradas automáticamente** como probabilidades reales de impago. El threshold 0.5 es provisional.
5. El test permanece reservado. **Todavía no existe un resultado final**; los números anteriores corresponden solo a validación.

## Qué cambio requiere el repositorio

**Ningún cambio a `src/splits.py`, `scripts/freeze_splits.py` o al contrato v4.** La decisión consolida el proceso actual. Versionar este documento y la evidencia de ablación; no reemplazar los resultados históricos del Avance 1.
