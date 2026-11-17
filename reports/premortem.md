# Premortem — evaluación final (Grupo 9)

**Supuesto:** entregamos y perdemos puntos. ¿Qué habrá fallado?

| Riesgo (prioridad) | Señal temprana | Control preventivo | Responsable de validación |
|---|---|---|---|
| **P0 — test inexistente o contaminado** | el notebook usa hasta jun-2014, la propuesta promete 2014H2 | elegir y congelar ventanas *después* del informe de conteos; un único split y test bloqueado | Valentino |
| **P0 — sesgo por censura** | muchos préstamos 2016-18 Current; los terminales recientes son atípicos | contrastar por año y plazo, justificar cohorte conservadora; prohibir all_final para training automático | Nicolás + Valentino |
| **P0 — leakage temporal/crediticio** | ROC-AUC cercano a 1 o cambios enormes al añadir variables | solo whitelist G1/G2, auditoría G3, transformaciones fit solo en train | Rubí |
| **P1 — baseline e informe no reproducibles** | notebook Colab tiene rutas absolutas, salidas no en repo | CLI desde raíz, requisitos, tests y manifest; reejecutar sobre CSV exacto | Valentino |
| **P1 — se optimiza usando el test** | se consultan métricas de 2014H2 al elegir hiperparámetros | validación para comparar; test solo una vez al final | Nicolás + Kris |
| **P1 — modelos no comparables** | train/valid distintos, features distintas sin registrar | mismo dataset/split/semilla/métricas, modelos en archivos separados | Nicolás + Kris |
| **P1 — casi todo el CSV queda fuera sin explicación** | 287,640 vs 2,260,701 sin tabla por estado/cohorte | ejecutar `python -m scripts.audit_dataset`, conservar desglose y razón de exclusión | Valentino + Nicolás |
| **P2 — métricas engañosas por desbalance** | solo accuracy o F1 sin tasa base | ROC-AUC, PR-AUC, matriz de confusión y prevalencia; umbral fijado usando validación | Kris |
| **P2 — resultados difíciles de interpretar** | AUC sin errores por segmento | tasas y errores por propósito/FICO/estado, con tamaños de grupo | Rubí |
| **P2 — carreras en Git** | varios editan `01_exploracion_inicial.ipynb` | congelar Avance 1, separar src + notebooks numerados y PR por rama | Todo el equipo |

## Decisiones pendientes, no resolver con suposiciones

1. Ejecutar auditoría REAL, no sustituirla por los tamaños de otros repositorios.
2. Decidir objetivo exacto: probabilidad de desenlace terminal **entre originaciones maduras** o definición alternativa con horizonte fijo (no disponible directamente en esta tabla estática).
3. Comparar políticas `legacy`, `matured`, `matured_36` por volumen, clase y periodo antes de escoger el dataset final.
4. Fijar, documentar y bloquear ventanas train / valid / test. **El test 2014H2 prometido no está dentro del dataset legacy**.
5. Confirmar con el diccionario de datos el instante de disponibilidad de `funded_amnt`, `verification_status`, `pymnt_plan` y otras variables fronterizas. La whitelist restringida del baseline no las usa.
