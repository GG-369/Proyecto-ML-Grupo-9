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