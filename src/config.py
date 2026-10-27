"""Contrato común del problema; cambios requieren revisión del equipo."""
from pathlib import Path

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / 'data' / 'raw' / 'accepted_2007_to_2018Q4.csv.gz'
DEFAULT_AUDIT = PROJECT_ROOT / 'outputs' / 'audit_lending_club.json'

# Solo desenlaces terminales. No imputar etiquetas para Current/Late/Grace.
FINAL_GOOD = frozenset({'Fully Paid'})
FINAL_BAD = frozenset({'Charged Off', 'Default'})
FINAL_STATUSES = FINAL_GOOD | FINAL_BAD

# Hipótesis para el filtro conservador, no una fecha de snapshot verificada:
# Se trata el archivo 2018Q4 como observado a fines de diciembre de 2018.
SNAPSHOT_ASSUMED = '2018-12-31'
GRACE_MONTHS = 6

# La cohorte 'legacy' reproduce el Avance 1, no define el test final.
LEGACY_START = '2012-01-01'
LEGACY_END = '2014-06-30'
