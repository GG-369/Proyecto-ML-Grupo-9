"""Contrato temporal v4, exclusivamente para préstamos a 36 meses.

Se utiliza el snapshot retrospectivo de 2018Q4; NO es una simulación
prospectiva de etiquetas conocidas al inicio de cada cohorte.
"""
from pathlib import Path
import json
import pandas as pd

from src.features import select_feature_columns

CONTRACT_ID = 'lc_36m_2007_2015h1_v4'
TRAIN_END = pd.Timestamp('2014-07-01')
VALID_END = pd.Timestamp('2015-01-01')
TEST_END = pd.Timestamp('2015-07-01')
EXPECTED_COUNTS = {'train': 244_943, 'valid': 90_615, 'test': 120_790}
EXPECTED_TOTAL = 456_348


def label_windows(frame):
    """Asignar ventanas por issue_d, sin consultar el target."""
    dates = pd.to_datetime(frame['issue_d'], errors='coerce')
    result = pd.Series('outside', index=frame.index, dtype='string')
    result.loc[dates.lt(TRAIN_END)] = 'train'
    result.loc[dates.ge(TRAIN_END) & dates.lt(VALID_END)] = 'valid'
    result.loc[dates.ge(VALID_END) & dates.lt(TEST_END)] = 'test'
    return result


def verify_partition(data, partition):
    if data.empty:
        raise ValueError(f'{partition}: partición vacía')
    if not data['term'].eq(36).all():
        raise ValueError(f'{partition}: se detectaron préstamos diferentes de 36 meses')
    if data['id'].isna().any() or data['id'].astype(str).str.strip().eq('').any():
        raise ValueError(f'{partition}: IDs ausentes')
    if data['id'].astype(str).duplicated().any():
        raise ValueError(f'{partition}: IDs duplicados')
    if not data['target'].isin((0, 1)).all():
        raise ValueError(f'{partition}: target inválido')
    bounds = {
        'train': (pd.Timestamp('1900-01-01'), TRAIN_END),
        'valid': (TRAIN_END, VALID_END),
        'test': (VALID_END, TEST_END),
    }
    start, end = bounds[partition]
    dates = pd.to_datetime(data['issue_d'], errors='coerce')
    if not (dates.ge(start) & dates.lt(end)).all():
        raise ValueError(f'{partition}: fechas fuera del contrato')
    num, cat = select_feature_columns(False)
    missing = set(num + cat) - set(data.columns)
    if missing:
        raise ValueError(f'{partition}: faltan features G1: {sorted(missing)}')
    return True


def freeze_from_parts(data_dir, audit_path, output_dir):
    """Generar SOLO archivos train/valid. La partición test permanece sellada.

    Lee temporalmente el dataset preparado para comprobar que los conteos
    concuerden con la auditoría. Nunca escribe ni exporta predicciones de test.
    """
    data_dir = Path(data_dir)
    audit_path = Path(audit_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'El contrato ya existe: {output_dir}. No sobreescribir splits congelados.')
    if not audit_path.is_file():
        raise FileNotFoundError(f'Falta auditoría: {audit_path}')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    expected = audit['cohort_totals']['matured_36_final']['final']
    if expected != EXPECTED_TOTAL:
        raise ValueError(f'El snapshot no coincide: esperado {EXPECTED_TOTAL}; encontrado {expected}')
    paths = sorted(data_dir.glob('part-*.parquet'))
    if not paths:
        raise FileNotFoundError(f'No hay Parquets de matured_36 en {data_dir}')
    manifest_path = data_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Falta manifest.json en {data_dir}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('mode') != 'matured_36':
        raise ValueError('La cohorte preparada no es matured_36')
    if manifest.get('rows_exported') != EXPECTED_TOTAL:
        raise ValueError('El Parquet no coincide con los conteos auditados')

    frames = [pd.read_parquet(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    if len(df) != EXPECTED_TOTAL:
        raise ValueError(f'Filas inconsistentes: {len(df)} vs {EXPECTED_TOTAL}')
    if df['id'].isna().any() or df['id'].astype(str).duplicated().any():
        raise ValueError('IDs faltantes o duplicados en la cohorte completa')
    labels = label_windows(df)
    if labels.eq('outside').any():
        raise ValueError('Se detectaron registros fuera de las ventanas del contrato')
    split_data = {name: df.loc[labels.eq(name)].copy() for name in ('train','valid','test')}
    for name, part in split_data.items():
        verify_partition(part, name)
    counts = {name: len(part) for name, part in split_data.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f'Los tamaños cambiaron: {counts} != {EXPECTED_COUNTS}')
    for a,b in (('train','valid'),('train','test'),('valid','test')):
        if set(split_data[a]['id'].astype(str)) & set(split_data[b]['id'].astype(str)):
            raise ValueError(f'IDs repetidos entre {a} y {b}')

    output_dir.mkdir(parents=True, exist_ok=False)
    split_data['train'].to_parquet(output_dir/'train.parquet', index=False)
    split_data['valid'].to_parquet(output_dir/'valid.parquet', index=False)
    # Nunca escribir test.parquet ni sumar su target en la selección del modelo.
    output = {
        'contract_id': CONTRACT_ID,
        'source': manifest.get('source_file'),
        'mode': 'matured_36',
        'counts': counts,
        'windows': {
            'train': '2007-01-01 <= issue_d < 2014-07-01',
            'valid': '2014-07-01 <= issue_d < 2015-01-01',
            'test_sealed': '2015-01-01 <= issue_d < 2015-07-01',
        },
        'published_artifacts': ['train.parquet','valid.parquet'],
        'test_note': 'Reservado; NO exportado, NO usar para elegir modelos',
        'snapshot_note': 'Corte 2018Q4 asumido; validación retrospectiva, no simulación de etiquetas disponibles en tiempo real',
    }
    (output_dir/'contract.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    return output
