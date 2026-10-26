# Datos de Lending Club

Guardar `accepted_2007_to_2018Q4.csv.gz` en `data/raw/` tras descargarlo de
[Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
El archivo de datos y las credenciales **no se suben** a Git.

- `data/raw/`: CSV original externo, sin alterar.
- `data/processed/`: fragmentos Parquet reproducibles, generados por `scripts/prepare_data.py`.
- `data/cache/`: opcional, temporal.

El *chunk size* controla memoria, no significa muestreo. Revisar primero
`outputs/audit_<nombre>.json`: no fijar nueva ventana ni declarar resultados del
corpus completo antes de correr esa auditoría con el CSV original.

Todos los recuentos de este proyecto se basan solo en préstamos aceptados; los
rechazados carecen de `loan_status` comparable y no se pueden etiquetar como
incumplimiento.
