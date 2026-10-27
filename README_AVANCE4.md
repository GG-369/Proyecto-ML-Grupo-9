# Cómo aplicar Avance 4 (sobre el Avance 3 ya instalado)

1. Copiar archivos de este ZIP a la raíz del repo y conservar el `.git` existente.
2. Asegurar que `outputs/audit_cohort_detail.json` existe (ya lo generaste).
3. Ejecutar `python -m pytest -q`.
4. Ejecutar `python -m scripts.prepare_data --mode matured_36` (requiere el CSV original en `data/raw/`).
5. Ejecutar `python -m scripts.freeze_splits` (genera train + valid y reserva test).
6. Ejecutar `python -m scripts.train_baseline_v4` (genera métricas y scores solo de validación).

En PowerShell de Windows reemplazar `python` por `.\.venv\Scripts\python.exe`.

Si `data/processed/matured_36/` ya contiene parquet, NO repitas prepare_data a menos que vayas a regenerarlo expresamente; continúa en el paso 5.

**Importante**: la separación v4 es retrospectiva por fecha de emisión y no prueba que las etiquetas fueran conocidas al inicio de cada ventana. El baseline del Avance 1 no es comparable numéricamente sin reentrenamiento bajo el contrato v4.
