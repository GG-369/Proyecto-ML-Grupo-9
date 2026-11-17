"""Entrena DecisionTree y RandomForest con el contrato v4; nunca usa test."""
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
from src.models_tree import (
    make_decision_tree,
    make_random_forest,
    top_decile_capture,
)
from src.splits import CONTRACT_ID, verify_partition

INNER_SPLIT = pd.Timestamp('2013-07-01')

CONFIGS = {
    'decision_tree': [
        {'max_depth': 6, 'min_samples_leaf': 100, 'class_weight': 'balanced'},
        {'max_depth': 12, 'min_samples_leaf': 100, 'class_weight': 'balanced'},
        {'max_depth': None, 'min_samples_leaf': 100, 'class_weight': 'balanced'},
        {'max_depth': 12, 'min_samples_leaf': 500, 'class_weight': 'balanced'},
    ],
    'random_forest': [
        {'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 100,
         'max_features': 'sqrt', 'class_weight': 'balanced_subsample'},
        {'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 300,
         'max_features': 'sqrt', 'class_weight': 'balanced_subsample'},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 100,
         'max_features': 'sqrt', 'class_weight': 'balanced_subsample'},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 300,
         'max_features': 'sqrt', 'class_weight': 'balanced_subsample'},
    ],
}
MODEL_FACTORIES = {
    'decision_tree': make_decision_tree,
    'random_forest': make_random_forest,
}


def load_contract_data(data_dir):
    """Cargar y comprobar únicamente train/valid del contrato conocido."""
    data_dir = Path(data_dir)
    contract_path = data_dir / 'contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    if contract.get('contract_id') != CONTRACT_ID:
        raise ValueError('Contrato desconocido; no entrenar sobre datos distintos')
    if (data_dir / 'test.parquet').exists():
        raise RuntimeError('test.parquet no debe existir en la carpeta de splits')

    train = pd.read_parquet(data_dir / 'train.parquet')
    valid = pd.read_parquet(data_dir / 'valid.parquet')
    verify_partition(train, 'train')
    verify_partition(valid, 'valid')
    if (len(train) != contract['counts']['train'] or
            len(valid) != contract['counts']['valid']):
        raise ValueError('El contrato no coincide con los archivos de datos')
    if set(train['id'].astype(str)) & set(valid['id'].astype(str)):
        raise ValueError('Filtración: IDs duplicados entre train y valid')
    if train['issue_d'].max() >= valid['issue_d'].min():
        raise ValueError('Las particiones train/valid no son estrictamente temporales')
    return train, valid


def search_configurations(train, model_name, include_g2=False):
    """Elegir hiperparámetros solo con una subvalidación temporal de train."""
    if model_name not in CONFIGS:
        raise ValueError(f'Modelo desconocido: {model_name}')
    numeric, categorical = select_feature_columns(include_g2)
    columns = numeric + categorical
    dates = pd.to_datetime(train['issue_d'], errors='coerce')
    inner_train = train.loc[dates < INNER_SPLIT]
    inner_valid = train.loc[dates >= INNER_SPLIT]
    if (inner_train.empty or inner_valid.empty or
            inner_train['target'].nunique() != 2 or
            inner_valid['target'].nunique() != 2):
        raise ValueError('La subvalidación temporal debe contener ambas clases en ambos lados')
    if set(inner_train['id'].astype(str)) & set(inner_valid['id'].astype(str)):
        raise ValueError('Filtración de IDs en la subvalidación temporal')

    rows = []
    factory = MODEL_FACTORIES[model_name]
    for config_id, params in enumerate(CONFIGS[model_name]):
        model = factory(include_g2, **params)
        started = time.perf_counter()
        model.fit(inner_train[columns], inner_train['target'])
        scores = model.predict_proba(inner_valid[columns])[:, 1]
        rows.append({
            'config_id': config_id,
            **params,
            'inner_roc_auc': float(roc_auc_score(inner_valid['target'], scores)),
            'fit_and_score_seconds': round(time.perf_counter() - started, 1),
        })
    return pd.DataFrame(rows).sort_values(
        'inner_roc_auc', ascending=False, kind='stable'
    ).reset_index(drop=True)


def fit_and_evaluate(train, valid, model_name, params, include_g2=False):
    """Reentrenar en todo train y evaluar una sola vez en valid."""
    numeric, categorical = select_feature_columns(include_g2)
    columns = numeric + categorical
    model = MODEL_FACTORIES[model_name](include_g2, **params)
    started = time.perf_counter()
    model.fit(train[columns], train['target'])
    fit_seconds = time.perf_counter() - started
    scores = model.predict_proba(valid[columns])[:, 1]
    metrics = evaluate_binary(valid['target'].to_numpy(), scores)
    metrics['top_decile'] = top_decile_capture(
        valid['target'].to_numpy(), scores, valid['id'].to_numpy()
    )
    metrics['fit_seconds'] = round(fit_seconds, 1)
    return metrics, scores


def run(data_dir, out_dir, with_g2=False):
    train, valid = load_contract_data(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = [('g1', False)]
    if with_g2:
        feature_sets.append(('g1_g2', True))

    result = {
        'contract_id': CONTRACT_ID,
        'train_n': len(train),
        'valid_n': len(valid),
        'seed': RANDOM_STATE,
        'inner_validation': (
            f'fit issue_d < {INNER_SPLIT.date()}, '
            'score el resto de train; selección por ROC-AUC'
        ),
        'versions': {
            'python': platform.python_version(),
            'sklearn': sklearn.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
        },
        'results': {},
    }
    predictions = pd.DataFrame({
        'id': valid['id'].astype('string'),
        'issue_d': pd.to_datetime(valid['issue_d']),
        'target': valid['target'].astype(int),
    })

    for feature_tag, include_g2 in feature_sets:
        for model_name in CONFIGS:
            print(f'== Búsqueda {model_name} {feature_tag} ==')
            search = search_configurations(train, model_name, include_g2)
            search.to_csv(
                out_dir / f'search_{model_name}_nicolas_{feature_tag}.csv',
                index=False,
            )
            best = search.iloc[0]
            params = {
                name: (None if pd.isna(best[name]) else best[name])
                for name in CONFIGS[model_name][0]
            }
            for name in ('max_depth', 'min_samples_leaf', 'n_estimators'):
                if name in params and pd.notna(params[name]):
                    params[name] = int(params[name])
            if params.get('max_depth') is not None:
                params['max_depth'] = int(params['max_depth'])

            metrics, scores = fit_and_evaluate(
                train, valid, model_name, params, include_g2
            )
            result['results'][f'{model_name}_{feature_tag}'] = {
                'params': params,
                'inner_roc_auc': float(best['inner_roc_auc']),
                **metrics,
            }
            predictions[f'score_{model_name}_{feature_tag}'] = scores

    (out_dir / 'metrics_tree_nicolas_v4.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    predictions.to_csv(
        out_dir / 'predictions_valid_tree_nicolas.csv', index=False
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description='DecisionTree y RandomForest para validación v4; nunca test'
    )
    parser.add_argument(
        '--data-dir',
        default=str(PROJECT_ROOT / 'data' / 'splits' / 'lc_36m_v4'),
    )
    parser.add_argument('--out-dir', default=str(PROJECT_ROOT / 'outputs'))
    parser.add_argument(
        '--with-g2',
        action='store_true',
        help='ejecuta G1+G2 como experimento separado además de G1',
    )
    args = parser.parse_args()
    result = run(args.data_dir, args.out_dir, args.with_g2)
    for name, metrics in result['results'].items():
        print(
            f'{name}: ROC-AUC={metrics["roc_auc"]:.4f} '
            f'PR-AUC={metrics["pr_auc"]:.4f} F1={metrics["f1"]:.4f} '
            f'top10%-capture={metrics["top_decile"]["capture"]:.3f}'
        )
    print('Evaluación solo en VALIDACIÓN; test permanece reservado.')


if __name__ == '__main__':
    main()
