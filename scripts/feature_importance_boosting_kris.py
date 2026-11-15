"""Importancia de variables para HGB (Kris), G1 y G1+G2. SOLO train/valid; nunca test.

Reentrena el pipeline de HistGradientBoostingClassifier con los hiperparámetros
ganadores ya documentados en `outputs/metrics_boosting_kris_v4.json` (sección 3
del reporte), usando TODO train.parquet. Calcula permutation importance con
`sklearn.inspection.permutation_importance` sobre la validación congelada,
usando ROC-AUC como métrica, varias repeticiones y semilla 42.

No se usa el test reservado en ningún momento.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from src.config import PROJECT_ROOT, RANDOM_STATE
from src.features import select_feature_columns
from src.models_boosting import make_hgb
from src.splits import CONTRACT_ID, verify_partition

N_REPEATS = 10


def load_contract_data(data_dir):
    data_dir = Path(data_dir)
    contract = json.loads((data_dir / 'contract.json').read_text(encoding='utf-8'))
    if contract.get('contract_id') != CONTRACT_ID:
        raise ValueError('Contrato desconocido; no entrenar sobre datos distintos')
    if (data_dir / 'test.parquet').exists():
        raise RuntimeError('test.parquet no debe existir en la carpeta de splits')
    train = pd.read_parquet(data_dir / 'train.parquet')
    valid = pd.read_parquet(data_dir / 'valid.parquet')
    verify_partition(train, 'train')
    verify_partition(valid, 'valid')
    if len(train) != contract['counts']['train'] or len(valid) != contract['counts']['valid']:
        raise ValueError('El contrato no coincide con los archivos de datos')
    if set(train['id'].astype(str)) & set(valid['id'].astype(str)):
        raise ValueError('Filtración: IDs duplicados entre train y valid')
    return train, valid


def load_winning_params(metrics_path, tag):
    """Lee los hiperparámetros ganadores ya fijados por la búsqueda aleatoria."""
    metrics = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    params = dict(metrics['results'][f'hgb_{tag}']['params'])
    max_iter = params.pop('max_iter')
    return params, max_iter


def run_group(train, valid, include_g2, tag, metrics_path, n_repeats, seed):
    num, cat = select_feature_columns(include_g2)
    cols = num + cat
    params, max_iter = load_winning_params(metrics_path, tag)
    print(f'== {tag}: reentrenando con params ganadores {params} max_iter={max_iter} ==')

    model = make_hgb(include_g2, max_iter=max_iter, **params)
    t0 = time.time()
    model.fit(train[cols], train['target'])
    fit_s = time.time() - t0
    print(f'  fit_seconds={fit_s:.1f}')

    t0 = time.time()
    result = permutation_importance(
        model, valid[cols], valid['target'],
        scoring='roc_auc', n_repeats=n_repeats, random_state=seed, n_jobs=-1,
    )
    perm_s = time.time() - t0
    print(f'  permutation_importance_seconds={perm_s:.1f}')

    g1_cols = set(select_feature_columns(False)[0] + select_feature_columns(False)[1])
    rows = []
    for i, col in enumerate(cols):
        rows.append({
            'feature': col,
            'group': 'G1' if col in g1_cols else 'G2',
            'importance_mean': float(result.importances_mean[i]),
            'importance_std': float(result.importances_std[i]),
        })
    rows.sort(key=lambda r: r['importance_mean'], reverse=True)
    return {
        'tag': tag,
        'include_g2': include_g2,
        'params': {**params, 'max_iter': max_iter},
        'fit_seconds': round(fit_s, 1),
        'permutation_importance_seconds': round(perm_s, 1),
        'n_repeats': n_repeats,
        'seed': seed,
        'scoring': 'roc_auc',
        'n_valid': len(valid),
        'importances': rows,
    }


def run(data_dir, out_dir, metrics_path, n_repeats, seed):
    train, valid = load_contract_data(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'contract_id': CONTRACT_ID, 'train_n': len(train), 'valid_n': len(valid),
        'seed': seed, 'n_repeats': n_repeats, 'scoring': 'roc_auc',
        'note': ('Permutation importance sobre la validación congelada, con los '
                 'hiperparámetros ya elegidos por la búsqueda aleatoria (sección 3 '
                 'del reporte). No se usó el test reservado.'),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__},
        'results': {},
    }
    for tag, g2 in (('g1', False), ('g1_g2', True)):
        result['results'][tag] = run_group(train, valid, g2, tag, metrics_path, n_repeats, seed)
    out_path = out_dir / 'feature_importance_boosting_kris.json'
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                         encoding='utf-8')
    print(f'Guardado en {out_path}')
    return result


def main():
    ap = argparse.ArgumentParser(description='Permutation importance HGB (Kris); nunca test')
    ap.add_argument('--data-dir', default=str(PROJECT_ROOT / 'data' / 'splits' / 'lc_36m_v4'))
    ap.add_argument('--out-dir', default=str(PROJECT_ROOT / 'outputs'))
    ap.add_argument('--metrics-path',
                     default=str(PROJECT_ROOT / 'outputs' / 'metrics_boosting_kris_v4.json'))
    ap.add_argument('--n-repeats', type=int, default=N_REPEATS)
    ap.add_argument('--seed', type=int, default=RANDOM_STATE)
    a = ap.parse_args()
    r = run(a.data_dir, a.out_dir, a.metrics_path, a.n_repeats, a.seed)
    for tag, res in r['results'].items():
        top5 = res['importances'][:5]
        print(f'-- {tag} top 5 --')
        for row in top5:
            print(f"  {row['feature']}: mean={row['importance_mean']:.4f} std={row['importance_std']:.4f}")
    print('Solo VALIDACIÓN; test permanece reservado.')


if __name__ == '__main__':
    main()
