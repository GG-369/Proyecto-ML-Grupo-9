"""Ablación predefinida: ¿conviene entrenar también con 2007–2011?

Ejecutar SOLO con los splits congelados v4; evalúa sobre validación,
NUNCA lee/parcela/exporta el conjunto test reservado.

Uso: python -m scripts.compare_training_windows_v41
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.config import PROJECT_ROOT
from src.evaluate import evaluate_binary
from src.features import select_feature_columns
from src.splits import CONTRACT_ID, verify_partition
from src.train import make_baseline

START_RECENT = pd.Timestamp('2012-01-01')
EXPECTED_RECENT_TRAIN = 215_847


def compare_windows(train: pd.DataFrame, valid: pd.DataFrame):
    """Comparar dos políticas de entrenamiento sobre idéntica validación."""
    verify_partition(train, 'train')
    verify_partition(valid, 'valid')
    if set(train['id'].astype(str)) & set(valid['id'].astype(str)):
        raise ValueError('IDs solapados entre entrenamiento y validación.')
    recent = train.loc[pd.to_datetime(train['issue_d']) >= START_RECENT].copy()
    if recent.empty or len(recent) == len(train):
        raise ValueError('La comparación necesita registros anteriores y posteriores a 2012.')
    if not recent['target'].isin([0, 1]).all() or recent['target'].nunique() != 2:
        raise ValueError('El entrenamiento reciente no tiene ambas clases.')
    results = {}
    for name, cohort in [('full_2007_2014h1', train), ('recent_2012_2014h1', recent)]:
        row = {'train_n': int(len(cohort)), 'models': {}}
        for model_name, use_g2 in [('logreg_g1', False), ('logreg_g1_g2', True)]:
            num, cat = select_feature_columns(use_g2)
            m = make_baseline(include_g2=use_g2)
            m.fit(cohort[num + cat], cohort['target'])
            score = m.predict_proba(valid[num + cat])[:, 1]
            row['models'][model_name] = evaluate_binary(valid['target'].to_numpy(), score)
        results[name] = row
    deltas = {}
    for model_name in ('logreg_g1', 'logreg_g1_g2'):
        full = results['full_2007_2014h1']['models'][model_name]
        recent = results['recent_2012_2014h1']['models'][model_name]
        deltas[model_name] = {
            'delta_roc_auc_full_minus_recent': full['roc_auc'] - recent['roc_auc'],
            'delta_pr_auc_full_minus_recent': full['pr_auc'] - recent['pr_auc'],
        }
    return {'contract_id': CONTRACT_ID, 'evaluation_split': 'valid_2014h2',
            'valid_n': len(valid), 'cohorts': results, 'comparison': deltas,
            'note': 'Selección SOLO en validación; ninguna prueba sobre test. '
                    'Comparación de políticas, no efecto causal de la antigüedad.'}


def run(data_dir, output):
    data_dir = Path(data_dir)
    contract = json.loads((data_dir / 'contract.json').read_text(encoding='utf-8'))
    if contract.get('contract_id') != CONTRACT_ID:
        raise ValueError('El contrato del dataset no coincide con v4.')
    train = pd.read_parquet(data_dir / 'train.parquet')
    valid = pd.read_parquet(data_dir / 'valid.parquet')
    if len(train) != contract['counts']['train'] or len(valid) != contract['counts']['valid']:
        raise ValueError('Inconsistencia entre contrato y Parquets.')
    if len(train) != 244_943 or len(valid) != 90_615:
        raise ValueError('Conteos v4 inesperados; no ejecutar una comparación incompatible.')
    recent_n = int((pd.to_datetime(train['issue_d']) >= START_RECENT).sum())
    if recent_n != EXPECTED_RECENT_TRAIN:
        raise ValueError(f'Entrenamiento reciente inesperado: {recent_n} != {EXPECTED_RECENT_TRAIN}.')
    result = compare_windows(train, valid)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description='Comparar políticas de train sin abrir test.')
    parser.add_argument('--data-dir', default=str(PROJECT_ROOT / 'data' / 'splits' / 'lc_36m_v4'))
    parser.add_argument('--output', default=str(PROJECT_ROOT / 'outputs' / 'comparison_training_windows_v41.json'))
    args = parser.parse_args()
    result = run(args.data_dir, args.output)
    for model, delta in result['comparison'].items():
        print(model, 'ROC-AUC full - recent:', round(delta['delta_roc_auc_full_minus_recent'], 6),
              'PR-AUC full - recent:', round(delta['delta_pr_auc_full_minus_recent'], 6))
    print('Comparación terminada sobre VALIDACIÓN; test reservado.')


if __name__ == '__main__':
    main()
