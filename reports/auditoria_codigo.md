# Auditoría del estado inicial — 26 de octubre de 2026

## Alcance

- Fuente inspeccionada: rama `main`, commit `be08e6323f016ce71feef742a3bb5fcb55562ce5`.
- Archivos verificados: `README.md`, `proposal.md`, `notebooks/01_exploracion_inicial.ipynb`, `reports/tabla_variables.csv`, `outputs/metrics_baseline.json` y rúbrica del curso.
- **No estaba incluido el CSV original**. Los tamaños del Avance 1 son cifras reportadas por el equipo; no se recalcularon.

## Hechos confirmados por el código

1. `chunksize=200_000` es lectura incremental, **no un recorte a 200 000**.
2. El filtro restringe originación a **2012-01-01 hasta 2014-06-30** y estado final a `Fully Paid`, `Charged Off` o `Default`.
3. El `target` es 1 para `Charged Off` / `Default`, 0 para `Fully Paid`.
4. Solo se usan **14 predictores numéricos G1 y 4 categóricos G1** en el baseline. La segunda versión suma `int_rate`, `installment` y `grade` (G2). No se usan todas las G1 listadas en `tabla_variables.csv`.
5. Las variables G3 de postpago se introducen **intencionalmente solo en el experimento demostrativo de leakage**, no en el baseline oficial.
6. Se entrena con 2012-2013 y se valida con 2014H1. El *test 2014H2* descrito en la propuesta **no está dentro de la selección ni separado en código**.
7. Los valores de AUC guardados se conservan como evidencia del Avance 1; el nuevo pipeline aún no reproduce resultados hasta usar el CSV de Kaggle.
8. El notebook instala y ejecuta con rutas fijas `/content/` de Colab, por lo que no se reproduce directamente en el VS Code local sin adaptación.
9. La tabla G0/G1/G2/G3 clasifica variables, pero la disponibilidad real al instante de decisión de algunos atributos no está validada con el diccionario de datos.

## Discrepancias / alertas

- No llamar al filtro de fecha **limpieza que eliminó el 87%**: es **recorte de cohorte** deliberado, al que sigue el filtro por estado.
- No calificar como *default* a `Current` o `Late`: falta su etiqueta final; excluirlos sin revisar observación y plazo introduce **sesgo de selección por tiempo de seguimiento**, no solo pérdida de datos.
- No añadir automáticamente todos los `G1` del archivo CSV. `funded_amnt`, `funded_amnt_inv`, `pymnt_plan`, `disbursement_method`, `initial_list_status` y algunas variables conjuntas requieren revisión explícita de cuándo fueron conocidas. El baseline actual es conservador.
- La fecha **2018Q4** figura en el nombre del fichero, pero no certifica por sí sola el instante exacto en que se actualizaron todos los `loan_status`. `matured` y `matured_36` son políticas candidatas con **supuesto** de corte, no un estándar validado.
- Tres familias requeridas por rúbrica: regresión logística, árboles/random forest y boosting; ninguna puede compararse honestamente con distintas particiones.

## Siguiente hito verificable

Ejecutar `python -m scripts.audit_dataset` sobre el CSV original y revisar `outputs/audit_lending_club.json`. A partir de esa salida: (a) elegir cohorte/objetivo, (b) fijar tres ventanas temporales comparables, (c) **congelar** test, y (d) recién después entrenar los modelos restantes. Esto evita inventar tamaños, tasas o mejoras de modelos.
