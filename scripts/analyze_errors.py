"""Análisis reproducible de errores de validación; jamás consulta test."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import evaluate_binary


def summarize(frame, score_column, threshold=0.5):
    score = frame[score_column].to_numpy(dtype=float)
    y = frame['target'].to_numpy(dtype=int)
    metrics = evaluate_binary(y, score, threshold)
    pred = score >= threshold
    month = pd.to_datetime(frame['issue_d'], errors='raise').dt.to_period('M').astype(str)
    per_month = []
    for label, idx in month.groupby(month).groups.items():
        positions = frame.index.get_indexer(idx)
        true = y[positions]
        guessed = pred[positions]
        per_month.append({
            'month': str(label), 'n': int(len(true)),
            'prevalence': float(true.mean()),
            'false_positives': int(((true == 0) & guessed).sum()),
            'false_negatives': int(((true == 1) & ~guessed).sum()),
        })
    # Cuantiles por ranking, NO calibración: promedios descriptivos.
    rank = pd.Series(score).rank(method='first', pct=True).to_numpy()
    decile = np.minimum((rank * 10).astype(int), 9)
    bins = []
    for i in range(10):
        mask = decile == i
        bins.append({'decile_low_to_high': i + 1, 'n': int(mask.sum()),
                     'mean_score': float(score[mask].mean()),
                     'actual_default_rate': float(y[mask].mean())})
    return {'metrics_at_threshold': metrics, 'by_issue_month': per_month,
            'score_deciles': bins}


def run(predictions, output):
    df = pd.read_csv(predictions, dtype={'id': 'string'})
    mandatory = {'id', 'issue_d', 'target'}
    if not mandatory.issubset(df) or df['id'].isna().any() or df['id'].duplicated().any():
        raise ValueError('Faltan columnas obligatorias o hay IDs duplicados.')
    columns = [c for c in df if c.startswith('score_')]
    if not columns:
        raise ValueError('No hay columnas score_ en las predicciones.')
    if df.isna().any().any():
        raise ValueError('Hay valores faltantes en predicciones.')
    df = df.reset_index(drop=True)
    result = {'source': Path(predictions).name, 'partition': 'valid',
              'threshold': 0.5, 'n': len(df),
              'models': {col: summarize(df, col) for col in columns}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def segment_report(predictions, valid_path, score_column='score_random_forest_g1'):
    """Unir predicciones a validación local y tabular errores por bandas fijas."""
    pred = pd.read_csv(predictions, dtype={'id': 'string'})
    valid = pd.read_parquet(valid_path)[['id', 'target', 'fico_avg', 'annual_inc']].copy()
    valid['id'] = valid['id'].astype('string')
    if pred['id'].duplicated().any() or valid['id'].duplicated().any():
        raise ValueError('IDs duplicados; la unión no es 1:1.')
    joined = pred.merge(valid, on='id', how='inner', validate='one_to_one',
                        suffixes=('_pred', '_valid'))
    if len(joined) != len(pred) or not (joined['target_pred'] == joined['target_valid']).all():
        raise ValueError('Las predicciones y valid.parquet no coinciden en IDs/etiquetas.')
    if score_column not in joined:
        raise ValueError('Score solicitado no disponible.')
    joined['fico_band'] = pd.cut(pd.to_numeric(joined['fico_avg'], errors='coerce'),
                                 bins=[-np.inf, 670, 740, np.inf],
                                 labels=['<670', '670-739', '>=740'], right=False)
    joined['income_band'] = pd.cut(pd.to_numeric(joined['annual_inc'], errors='coerce'),
                                   bins=[-np.inf, 40000, 100000, np.inf],
                                   labels=['<40k', '40k-99k', '>=100k'], right=False)
    result = {'score': score_column, 'n': len(joined), 'segments': {}}
    for field in ('fico_band', 'income_band'):
        rows = []
        for label, group in joined.groupby(field, observed=True):
            y = group['target_pred'].to_numpy(dtype=int)
            guessed = group[score_column].to_numpy(dtype=float) >= 0.5
            tp = int((guessed & (y == 1)).sum())
            fp = int((guessed & (y == 0)).sum())
            fn = int((~guessed & (y == 1)).sum())
            rows.append({'band': str(label), 'n': len(y), 'prevalence': float(y.mean()),
                         'false_positives': fp, 'false_negatives': fn,
                         'precision': tp / (tp + fp) if tp + fp else None,
                         'recall': tp / (tp + fn) if tp + fn else None})
        result['segments'][field] = rows
        result['segments'][field + '_missing'] = int(joined[field].isna().sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default='outputs/predictions_valid_tree_nicolas.csv')
    parser.add_argument('--output', default='outputs/error_analysis_tree_nicolas.json')
    parser.add_argument('--valid', help='valid.parquet local para segmentar por FICO/ingreso')
    parser.add_argument('--segment-output', default='outputs/error_segments_tree_nicolas.json')
    args = parser.parse_args()
    result = run(args.predictions, args.output)
    if args.valid:
        segments = segment_report(args.predictions, args.valid)
        path = Path(args.segment_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, item in result['models'].items():
        m = item['metrics_at_threshold']
        print(f'{name}: AUC={m["roc_auc"]:.4f}, FP={m["confusion_matrix"][0][1]}, '
              f'FN={m["confusion_matrix"][1][0]}')


if __name__ == '__main__':
    main()
