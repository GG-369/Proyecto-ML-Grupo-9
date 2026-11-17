"""Conteo exhaustivo en streaming y selección de cohortes sin crear etiquetas."""
from collections import Counter
import json
from pathlib import Path
import pandas as pd

from src.config import FINAL_STATUSES, FINAL_BAD, LEGACY_START, LEGACY_END
from src.features import RAW_NEEDED, derive_features

AUDIT_COLUMNS = ['issue_d', 'loan_status', 'term']


def parse_dates(values):
    return pd.to_datetime(values, format='%b-%Y', errors='coerce')


def parse_term(values):
    return pd.to_numeric(values.astype('string').str.extract(r'(\d+)', expand=False), errors='coerce')


def cohort_masks(frame):
    """Máscaras independientes; sus conteos NO se deben sumar entre sí.

    `matured`: la fecha de origen + plazo original + seis meses de margen
    no supera 2018Q4. Este proxy de completitud necesita validación en el
    archivo real; no corrige por sí solo todos los sesgos de censura.
    """
    issue = parse_dates(frame['issue_d'])
    term = parse_term(frame['term'])
    final = frame['loan_status'].isin(FINAL_STATUSES)
    legacy_date = issue.between(LEGACY_START, LEGACY_END)
    matured = ((term.eq(36) & issue.le(pd.Timestamp('2015-06-30'))) |
               (term.eq(60) & issue.le(pd.Timestamp('2013-06-30'))))
    matured36 = term.eq(36) & issue.le(pd.Timestamp('2015-06-30'))
    return {
        'legacy_date': legacy_date,
        'legacy_final': legacy_date & final,
        'all_final': final,
        'matured_date': matured,
        'matured_final': matured & final,
        'matured_36_date': matured36,
        'matured_36_final': matured36 & final,
    }


def audit_csv(path, chunksize=100_000):
    """Leer TODO el fichero sin descargarlo ni construir un dataframe gigante."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'No existe el CSV de origen: {path}')
    counts = Counter()
    statuses = Counter()
    years = Counter()
    year_and_status = Counter()
    for chunk in pd.read_csv(path, compression='infer', usecols=AUDIT_COLUMNS,
                             chunksize=chunksize, low_memory=False):
        issue = parse_dates(chunk['issue_d'])
        masks = cohort_masks(chunk)
        counts['raw_total'] += len(chunk)
        counts['issue_d_missing_or_invalid'] += int(issue.isna().sum())
        counts['loan_status_missing'] += int(chunk['loan_status'].isna().sum())
        counts['final_bad'] += int(chunk['loan_status'].isin(FINAL_BAD).sum())
        for key, mask in masks.items():
            counts[key] += int(mask.sum())
        statuses.update(chunk['loan_status'].fillna('<missing>').astype(str).value_counts().to_dict())
        years.update(issue.dt.year.fillna(-1).astype(int).astype(str).value_counts().to_dict())
        grouped = pd.DataFrame({'year': issue.dt.year.fillna(-1).astype(int).astype(str),
                                'status': chunk['loan_status'].fillna('<missing>').astype(str)})
        year_and_status.update({f'{year}|{status}': int(n) for (year, status), n
                                in grouped.value_counts().items()})
    result = {
        'source_file': path.name, 'source_bytes': path.stat().st_size,
        'counts': dict(sorted(counts.items())),
        'statuses': dict(sorted(statuses.items())),
        'by_issue_year': dict(sorted(years.items())),
        'by_year_and_status': dict(sorted(year_and_status.items())),
        'notes': [
            'Los conteos son exhaustivos: chunksize no es una muestra.',
            'Los candidatos all_final incluyen originaciones recientes y NO deben entrenarse sin auditoría de censura.',
            'matured usa plazo 36/60 meses y margen de 6 meses contra supuesto snapshot 2018Q4.',
            'Los cortes por plazo reducen censura, pero no garantizan seguimiento ni completitud perfecta.',
            'Las columnas de decisión jamás deben incluir loan_status ni variables de pagos futuros.',
        ],
    }
    return result


def write_audit(path, output, chunksize=100_000):
    result = audit_csv(path, chunksize)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def prepare_parquet_parts(path, output_dir, mode='legacy', chunksize=100_000):
    """Materializa un directorio de pequeños Parquets en streaming.

    Trabajar sobre una carpeta NUEVA/vacía: jamás mezcla cohortes antiguas.
    Solo exporta columnas de la allowlist (+ id, issue_d, target para auditoría).
    """
    if mode not in ('legacy', 'matured', 'matured_36'):
        raise ValueError('Modo inválido; all_final NO es un modo entrenable.')
    source = Path(path)
    output_dir = Path(output_dir)
    if not source.is_file(): raise FileNotFoundError(source)
    if output_dir.exists() and list(output_dir.glob('*.parquet')):
        raise FileExistsError(f'Carpeta ya contiene Parquets: {output_dir}. Usar otra carpeta.')
    output_dir.mkdir(parents=True, exist_ok=True)
    relevant_mask = {'legacy': 'legacy_final', 'matured': 'matured_final',
                     'matured_36': 'matured_36_final'}[mode]
    total, exported, part = 0, 0, 0
    for chunk in pd.read_csv(source, compression='infer', usecols=RAW_NEEDED,
                             chunksize=chunksize, low_memory=False,
                             dtype={'id': 'string'}):
        total += len(chunk)
        mask = cohort_masks(chunk)[relevant_mask]
        filtered = chunk.loc[mask].copy()
        if filtered.empty: continue
        filtered['target'] = filtered['loan_status'].isin(FINAL_BAD).astype('int8')
        filtered = derive_features(filtered)
        # Excluimos loan_status / earliest_cr_line / rangos FICO originales,
        # no se usarán para entrenar. Conservamos ID solo para auditar duplicados.
        final_columns = ['id', 'issue_d', 'target'] + sorted(set(
            ['loan_amnt', 'term', 'emp_length', 'annual_inc', 'dti',
             'delinq_2yrs', 'fico_avg', 'inq_last_6mths', 'open_acc',
             'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
             'hist_crediticio_anios', 'home_ownership',
             'verification_status', 'purpose', 'addr_state',
             'int_rate', 'installment', 'grade']))
        filtered = filtered[final_columns]
        filtered.to_parquet(output_dir / f'part-{part:05d}.parquet', index=False)
        exported += len(filtered); part += 1
    if not part:
        raise ValueError('La selección no encontró préstamos finales; revisar auditoría.')
    manifest = {'source_file': source.name, 'mode': mode,
                'raw_rows_scanned': total, 'rows_exported': exported,
                'parquet_parts': part, 'note': 'Conteos de este CSV local; no usar para otro snapshot.'}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
