# Predicción de incumplimiento en préstamos P2P (Lending Club)

Repositorio correspondiente a la **Entrega Previa (Avance 1)** del proyecto final de Machine Learning (CS3061 - UTEC).

## Descripción general
El objetivo de este avance es plantear el problema predictivo sobre un dataset real y complejo (Lending Club), demostrando control riguroso de *data leakage*, validación temporal, establecimiento de un baseline honesto y análisis exploratorio inicial. 

Toda la fundamentación teórica, estructuración de variables, métricas y planificación se encuentra detallada en el archivo [proposal.md](proposal.md).

## Estructura del repositorio
- `proposal.md`: Documento principal con la propuesta y alcance del proyecto (puntos 1 al 15).
- `README.md`: Guía de inicio y reproducción del repositorio.
- `requirements.txt`: Listado de dependencias técnicas de Python.
- `notebooks/01_exploracion_inicial.ipynb`: Notebook unificado que contiene la ingesta, el análisis exploratorio (EDA), las visualizaciones, pruebas de leakage y la ejecución del baseline.
- `reports/tabla_variables.csv`: Matriz de clasificación de atributos por grupos (G0, G1, G2, G3) para el control de fugas de datos.
- `outputs/metrics_baseline.json`: Archivo con los resultados métricos cuantitativos del baseline y modelos de referencia.

## Instrucciones de reproducción
1. El código de este avance se encuentra estructurado para ejecutarse íntegramente en **Google Colab**.
2. Asegúrate de tener una cuenta activa en [Kaggle](https://www.kaggle.com/) y genera un token de API (`Account -> Create New API Token`), el cual descargará un archivo llamado `kaggle.json`.
3. Abre el archivo `notebooks/01_exploracion_inicial.ipynb` en Google Colab.
4. Ejecuta las celdas de forma secuencial. La primera celda habilitará un widget para subir tu archivo `kaggle.json`, automatizando la descarga del dataset oficial (`wordsforthewise/lending-club`) directamente al entorno de ejecución.
5. El flujo de partición y modelado utiliza una semilla fija (`random_state=42`) para garantizar la total reproducibilidad de los resultados métricos obtenidos.

## Referencias
- Dataset: Lending Club Loan Data, vía Kaggle (`wordsforthewise/lending-club`).
- Curso de Machine Learning, UTEC (2026).

---

## Avance 2 en preparación (sin alterar Avance 1)

El notebook inicial y `outputs/metrics_baseline.json` se conservan íntegros.
Se preparó una estructura mínima en `src/`, `scripts/` y `tests/`; todavía **no
se ha reentrenado con el CSV completo**.

### Inicio en VS Code, PowerShell (desde la raíz del repositorio)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

Si PowerShell bloquea la activación, ejecutar directamente
`.\.venv\Scripts\python.exe -m pytest -q`.

Descargar el archivo desde Kaggle en `data/raw/accepted_2007_to_2018Q4.csv.gz`
(ver [`data/README.md`](data/README.md)). Nunca subir `kaggle.json`.

```powershell
# 1. Recuento de todos los registros, por estado y año (NO muestrea)
python -m scripts.audit_dataset

# 2. Reproducir cohorte histórica; archivos parquet por fragmentos
python -m scripts.prepare_data --mode legacy

# 3. Reentrenar baseline en validación (NO toca test)
python -m scripts.train_baseline
```

**Antes de escoger cohorte final:** contrastar `legacy`, `matured`, `matured_36`
con `outputs/audit_lending_club.json`. El modo `all_final` se cuenta en la
auditoría, pero se bloquea para preparación porque puede introducir sesgo de
censura en préstamos recientes. `matured` es una hipótesis de observación
basada en 2018Q4 + 6 meses, no una garantía de seguimiento completo.

**Test final pendiente:** la ventana 2014H2 prometida en `proposal.md` no se
incluye en `legacy`. No definirla ni evaluar sobre ella sin validar primero
las cohortes elegibles y fijar un protocolo temporal de tres conjuntos.

- Auditoría de código: [`reports/auditoria_codigo.md`](reports/auditoria_codigo.md)
- Premortem: [`reports/premortem.md`](reports/premortem.md)
- Asignación paralela: [`reports/plan_equipo.md`](reports/plan_equipo.md)
