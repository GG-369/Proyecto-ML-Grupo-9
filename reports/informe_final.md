# Informe integrado — estado de borrador verificable

**Grupo 9 · Predicción de incumplimiento Lending Club · 17/11/2026.** Las cifras siguientes son de **validación**, registradas por el equipo antes de esta integración. No se ha abierto el test para evaluar modelos. Este documento cubre las 15 secciones exigidas por la guía de Machine Learning; se debe completar la evaluación final antes de presentarlo como informe terminado.

## 1. Resumen ejecutivo

Para préstamos aceptados de 36 meses, se comparan un predictor constante, regresión logística, árbol, bosque y dos tipos de boosting. Entre los modelos G1, HistGradientBoosting (HGB) obtuvo ROC-AUC 0.6665 y PR-AUC 0.2329 en validación, frente a 0.6548/0.2231 de la logística. La diferencia es moderada. G1+G2 llega a ROC-AUC 0.6854/PR-AUC 0.2510 con HGB, pero aprovecha variables de la evaluación crediticia de Lending Club; se reporta por separado. No hay resultados finales de test.

## 2. Problema

Estimar el riesgo de que un préstamo ya aceptado termine en Charged Off/Default frente a Fully Paid. Una fila es un préstamo. Esta población no permite estimar resultados de solicitantes rechazados ni aprobar créditos de forma automática.

## 3. Dataset

Archivo accepted_2007_to_2018Q4.csv.gz de Lending Club vía Kaggle, documentado en data/README.md. La auditoría previa encontró 2 260 701 filas originales. La cohorte madura elegida de 36 meses contiene 456 348 casos terminales. El CSV bruto no se distribuye en Git; el corte 2018Q4 usado para madurez es una **hipótesis**, no una fecha de disponibilidad certificada.

## 4. Limpieza y preparación

src/data.py filtra estados terminales y préstamos a 36 meses; src/features.py convierte fechas, plazo, antigüedad laboral y números, deriva FICO promedio y años de historial crediticio. Faltantes numéricos se imputan con mediana; categóricos con moda; logística y árboles usan one-hot y los numéricos se escalan. HGB usa codificación ordinal de categorías, gestiona NaN y no necesita el mismo escalado. Cada ajuste aprendido ocurre dentro del Pipeline, ajustado solo sobre entrenamiento. No se justifica todavía variable por variable la magnitud de faltantes y outliers de la cohorte final: pendiente para el cierre.

## 5. Exploración

El notebook 01 documenta exploración y visualizaciones del Avance 1. Su población es distinta de la cohorte v4; no trasladar sus distribuciones sin repetirlas en la cohorte final. La auditoría y reports/contrato_dataset_v4.md describen tamaños y prevalencias (train 12.67 %, valid 14.13 %, test descriptivo 15.12 %). Falta un notebook limpio de EDA de v4 con gráficos interpretados.

## 6. Partición y validación

Por issue_d: train 2007–junio 2014 (244 943), valid julio–diciembre 2014 (90 615), test reservado enero–junio 2015 (120 790). scripts/freeze_splits.py exporta train/valid, nunca test.parquet, y comprueba IDs/contrato. Árboles y HGB seleccionan configuración mediante una subvalidación temporal dentro de train; la validación común mide los candidatos. **Advertencia central:** las etiquetas de préstamos originados en train pueden haberse conocido después del inicio de valid; es benchmark retrospectivo por fecha de origen, no backtest point-in-time auténtico. Ver reports/decision_train_v42.md.

## 7. Baselines

Dummy predice la clase mayoritaria: ROC-AUC 0.5000 y PR-AUC ≈ 0.1413 (la prevalencia). Regresión logística G1: ROC-AUC 0.6548 y PR-AUC 0.2231. El JSON de métricas v4 de logística no estaba versionado por la rama original, pero el script reproducible está en scripts/train_baseline_v4.py y los valores aparecen en los informes de modelos. Correrlo con el dataset local antes de fijar números finales.

## 8. Modelos entrenados

| Variables | Modelo | ROC-AUC valid | PR-AUC valid | Fuente |
|---|---|---:|---:|---|
| G1 | Dummy | 0.5000 | ≈0.1413 | scripts/train_baseline_v4.py; reporte de árboles |
| G1 | Logística | 0.6548 | 0.2231 | reports/resultados_arboles_nicolas.md |
| G1 | Árbol | 0.6323 | 0.2084 | outputs/metrics_tree_nicolas_v4.json |
| G1 | Bosque | 0.6560 | 0.2238 | outputs/metrics_tree_nicolas_v4.json |
| G1 | HGB | 0.6665 | 0.2329 | outputs/metrics_boosting_kris_v4.json |
| G1 | GB clásico | 0.6633 | 0.2301 | outputs/metrics_boosting_kris_v4.json |
| G1+G2 | Bosque | 0.6831 | 0.2478 | outputs/metrics_tree_nicolas_v4.json |
| G1+G2 | HGB | 0.6854 | 0.2510 | outputs/metrics_boosting_kris_v4.json |

El pipeline del árbol/forest no venía en su PR; esta integración lo reconstruyó con el preprocesador compartido. Sus archivos de predicción guardados reproducen las métricas publicadas al recalcularlas, pero **no se reentrenó** contra el CSV original aquí. Reproducir la corrida completa antes de afirmar igualdad exacta de artefactos. [Curvas ROC y precisión–recall G1](figures/validacion_arboles_g1.png) regeneradas desde el CSV guardado; muestran ordenación superior al azar, pero precisión modesta ante falsos positivos.

## 9. Hiperparámetros

Nicolás probó cuatro configuraciones por familia y por G1/G1+G2; eligió ROC-AUC en train partido temporalmente antes/después del 01/07/2013. G1: árbol profundidad 12/hoja 500; bosque 200 árboles, profundidad libre/hoja 100. Kris buscó 30 configuraciones de HGB con semilla 42, tasas, hojas, regularización y pesos, y fijó iteraciones usando subvalidación. El GB clásico fue una configuración fija; no se puede atribuir una superioridad algorítmica a esa comparación. Detalles y resultados de búsqueda: reports/resultados_arboles_nicolas.md y reports/resultados_boosting_kris.md.

## 10. Evaluación final

**Pendiente.** No hay métricas sobre test ni modelo final aprobado. No presentar validación como evaluación final. ROC-AUC mide ordenación entre pares de clases, no porcentaje de aciertos; PR-AUC resume precisión frente a recall. En G1, HGB tiene F1=0 al umbral 0.5 porque no marca casos positivos: seleccionar un umbral de operación requiere costes o capacidad de revisión y evaluación sin reutilizar test para ajustarlo.

## 11. Interpretación

Permutation importance para HGB G1 en validación identifica FICO promedio (caída ROC-AUC 0.0497), ingreso anual (0.0357), monto (0.0211). En G1+G2 predomina la tasa de interés (0.0589), que resume parte del juicio de Lending Club. No son efectos causales; variables correlacionadas pueden repartir importancia. Ver outputs/feature_importance_boosting_kris.json.

## 12. Análisis de errores

scripts/analyze_errors.py recalcula métricas, falsos positivos/negativos, prevalencia por mes y score/impago observado por decil desde el CSV versionado de Nicolás. outputs/error_analysis_tree_nicolas.json registra cuatro modelos. En Random Forest G1 al umbral 0.5 hubo 30 186 falsas alarmas y 4 996 impagos omitidos; precisión 20.54 %, recall 60.97 %. Con la opción --valid se unen IDs 1:1 a valid.parquet local y se calculan bandas fijas de FICO e ingreso; esa corrida todavía no puede ejecutarse aquí porque el Parquet no está en Git. Pendiente comprobar esos segmentos, discutir casos y sesgos; no inferir equidad a partir de los agregados actuales.

## 13. Riesgos éticos, sesgos y límites

Solo préstamos aceptados: sesgo de selección; variables como región/ingreso pueden asociarse con desigualdad; seguimiento y estados terminales tienen censura; snapshot supuesto y disponibilidad tardía de etiquetas impiden inferir funcionamiento real en 2014; puntuaciones no calibradas; umbral 0.5 arbitrario. G3 (recoveries, pagos, FICO posterior) queda excluido; G2 es información del prestamista que debe separarse según el momento de uso.

## 14. Trabajo futuro

Completar diagnóstico de calidad y EDA de v4, segmentación de errores y sensibilidad a umbral, protocolo de disponibilidad temporal, elección conjunta del modelo, evaluación única del holdout sin ajustar luego sobre él, informe con figuras legibles y exposición final.

## 15. Reproducción

Seguir README.md en orden, con CSV local original, Python/dependencias de requirements.txt, semilla 42. La integración ejecutó 22 pruebas con datos sintéticos y recalculó las métricas de cuatro modelos del CSV de predicciones de Nicolás. No hay garantía de reproducción integral de métricas sin el CSV original.

### Sobre la otra matriz adjunta

La matriz titulada «Matriz de Evaluación Entrega Final» que menciona Producer Direct, dashboard, notebooks 01_limpieza–04_dashboard y nombres Pedro/Jobeth/Valentino corresponde a **otro proyecto o curso**. No se usa para afirmar cumplimiento de este proyecto ML. Si el profesor confirma que también aplica, habrá que desarrollar ese producto por separado; este repositorio carece de sus notebooks y dashboard.
