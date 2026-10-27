"""Boosting (Kris) sobre el contrato v4. SOLO train/valid; nunca test.

Flujo por conjunto de variables (G1 obligatorio, G1+G2 experimento aparte):
1. Subvalidación temporal DENTRO de train: ajusta con issue_d < INNER_SPLIT y
   puntúa en el resto de train. Búsqueda aleatoria (semilla 42) y mejor número
   de iteraciones por configuración con staged_predict_proba.
2. Reentrena la mejor configuración con TODO train.
3. Evalúa UNA vez en la validación congelada (90,615) con evaluate_binary.
La validación congelada no interviene en la elección de hiperparámetros.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from src.config import PROJECT_ROOT, RANDOM_STATE
from src.evaluate import evaluate_binary
from src.features import select_feature_columns
from src.models_boosting import (make_gb_classic, make_hgb, sample_configs,
                                 top_decile_capture)
from src.splits import CONTRACT_ID, verify_partition

INNER_SPLIT = pd.Timestamp('2013-07-01')  # últimos 12 meses de train = subvalidación
MAX_ITER_CAP = 500
EVAL_EVERY = 10


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


def search_hgb(train, include_g2, n_iter):
    """Búsqueda aleatoria con subvalidación temporal dentro de train."""
    num, cat = select_feature_columns(include_g2)
    cols = num + cat
    dates = pd.to_datetime(train['issue_d'])
    inner_tr, inner_va = train[dates < INNER_SPLIT], train[dates >= INNER_SPLIT]
    if inner_tr.empty or inner_va.empty or inner_va['target'].nunique() < 2:
        raise ValueError('Subvalidación temporal inválida')
    rows = []
    for i, cfg in enumerate(sample_configs(n_iter)):
        t0 = time.time()
        pipe = make_hgb(include_g2, max_iter=MAX_ITER_CAP, **cfg)
        pipe.fit(inner_tr[cols], inner_tr['target'])
        Xv = pipe.named_steps['preprocess'].transform(inner_va[cols])
        best_auc, best_it = -1.0, 0
        for it, proba in enumerate(pipe.named_steps['model'].staged_predict_proba(Xv), 1):
            if it % EVAL_EVERY and it != MAX_ITER_CAP:
                continue
            auc = roc_auc_score(inner_va['target'], proba[:, 1])
            if auc > best_auc:
                best_auc, best_it = auc, it
        rows.append({'config_id': i, **cfg, 'best_iter': best_it,
                     'inner_roc_auc': best_auc, 'seconds': round(time.time() - t0, 1)})
        print(f'  [{i+1}/{n_iter}] inner ROC-AUC={best_auc:.4f} iter={best_it} {cfg}')
    return pd.DataFrame(rows).sort_values('inner_roc_auc', ascending=False)


def fit_eval(model, train, valid, include_g2):
    num, cat = select_feature_columns(include_g2)
    cols = num + cat
    t0 = time.time()
    model.fit(train[cols], train['target'])
    fit_s = time.time() - t0
    p = model.predict_proba(valid[cols])[:, 1]
    metrics = evaluate_binary(valid['target'].to_numpy(), p)
    metrics['top_decile'] = top_decile_capture(valid['target'].to_numpy(), p, 0.10)
    metrics['fit_seconds'] = round(fit_s, 1)
    return metrics, p


def run(data_dir, out_dir, n_iter, with_gb, only_g1):
    train, valid = load_contract_data(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'contract_id': CONTRACT_ID, 'train_n': len(train), 'valid_n': len(valid),
        'seed': RANDOM_STATE, 'n_search_configs': n_iter,
        'inner_validation': f'fit issue_d < {INNER_SPLIT.date()}, score el resto de train',
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__},
        'results': {},
    }
    preds = pd.DataFrame({'id': valid['id'].astype('string'),
                          'issue_d': pd.to_datetime(valid['issue_d']),
                          'target': valid['target'].astype(int)})
    for tag, g2 in (('g1', False),) if only_g1 else (('g1', False), ('g1_g2', True)):
        print(f'== Búsqueda HGB {tag} ==')
        search = search_hgb(train, g2, n_iter)
        search.to_csv(out_dir / f'search_boosting_kris_{tag}.csv', index=False)
        best = search.iloc[0]
        params = {k: (None if pd.isna(best[k]) else best[k]) for k in
                  ('learning_rate', 'max_leaf_nodes', 'max_depth', 'min_samples_leaf',
                   'l2_regularization', 'class_weight')}
        params['learning_rate'] = float(params['learning_rate'])
        params['l2_regularization'] = float(params['l2_regularization'])
        params['max_leaf_nodes'] = int(params['max_leaf_nodes'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        if params['max_depth'] is not None:
            params['max_depth'] = int(params['max_depth'])
        max_iter = int(best['best_iter'])
        print(f'Mejor config {tag}: {params} max_iter={max_iter}')
        model = make_hgb(g2, max_iter=max_iter, **params)
        metrics, p = fit_eval(model, train, valid, g2)
        result['results'][f'hgb_{tag}'] = {
            'params': {**params, 'max_iter': max_iter},
            'inner_roc_auc': float(best['inner_roc_auc']), **metrics}
        preds[f'score_hgb_{tag}'] = p
        if with_gb:
            print(f'== GradientBoosting clásico {tag} (lento) ==')
            metrics, p = fit_eval(make_gb_classic(g2), train, valid, g2)
            result['results'][f'gb_classic_{tag}'] = {
                'params': 'n_estimators=200, lr=0.1, max_depth=3, subsample=0.8 (fija, sin búsqueda)',
                **metrics}
            preds[f'score_gb_classic_{tag}'] = p
    (out_dir / 'metrics_boosting_kris_v4.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    preds.to_csv(out_dir / 'predictions_valid_boosting_kris.csv', index=False)
    return result


def main():
    ap = argparse.ArgumentParser(description='Boosting sobre validación v4; nunca test')
    ap.add_argument('--data-dir', default=str(PROJECT_ROOT / 'data' / 'splits' / 'lc_36m_v4'))
    ap.add_argument('--out-dir', default=str(PROJECT_ROOT / 'outputs'))
    ap.add_argument('--n-iter', type=int, default=20, help='configuraciones de la búsqueda aleatoria')
    ap.add_argument('--with-gb', action='store_true', help='añade GradientBoosting clásico (lento)')
    ap.add_argument('--only-g1', action='store_true')
    a = ap.parse_args()
    r = run(a.data_dir, a.out_dir, a.n_iter, a.with_gb, a.only_g1)
    for name, m in r['results'].items():
        td = m['top_decile']
        print(f'{name}: ROC-AUC={m["roc_auc"]:.4f} PR-AUC={m["pr_auc"]:.4f} '
              f'F1@0.5={m["f1"]:.4f} top10%-capture={td["capture"]:.3f} ({m["fit_seconds"]}s)')
    print('Solo VALIDACIÓN; test permanece reservado.')


if __name__ == '__main__':
    main()
