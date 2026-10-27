# Parche v4.2 — Cierre de política de entrenamiento y traspaso

Este ZIP **no cambia código de entrenamiento, particiones, métricas ni test**. Solo versiona la decisión sobre el uso de 2007–2011 y deja instrucciones de trabajo independientes para el equipo.

Copiar los archivos de `Proyecto-ML-Grupo-9/` al repositorio Git local, manteniendo `.git` y los archivos existentes. No extraer una copia de `.git` (el ZIP no incluye ninguna).

Archivos:
- `reports/decision_train_v42.md`: decisión, números verificados, limitaciones.
- `reports/comparison_training_windows_v41.json`: evidencia cuantitativa de la ablación ejecutada en validación.
- `reports/handoff_equipo_v42.md`: tareas, interfaces y criterios de integración.

Ejecutar desde PowerShell en la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git status --short
git add reports/decision_train_v42.md reports/handoff_equipo_v42.md reports/comparison_training_windows_v41.json README_V42.md
git commit -m "docs: fijar politica de entrenamiento y tareas por rama"
git push -u origin feat/infra-valentino
```

Si tienes otros cambios legítimos en tu rama que también deseas compartir, revísalos con `git status` y haz un commit separado. No es necesario ejecutar `prepare_data` ni `freeze_splits` otra vez.

Los otros integrantes necesitan datos locales: la data real **no va en Git**, cada uno debe prepararla desde Kaggle o acceder a archivos procesados por un canal privado autorizado. El conjunto test permanece sin usar.
