"""Entrena/evalúa EXCLUSIVAMENTE sobre train/valid del contrato v4.

No acepta argumento de test ni lee las filas de holdout.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.config import PROJECT_ROOT, RANDOM_STATE
from src.evaluate import evaluate_binary
from src.features import select_feature_columns
from src.splits import CONTRACT_ID, verify_partition
from src.train import make_baseline


def run(data_dir, output_metrics, output_predictions):
    data_dir=Path(data_dir)
    contract=json.loads((data_dir/'contract.json').read_text(encoding='utf-8'))
    if contract.get('contract_id') != CONTRACT_ID:
        raise ValueError('Contrato desconocido; no entrenar sobre datos distintos')
    train=pd.read_parquet(data_dir/'train.parquet')
    valid=pd.read_parquet(data_dir/'valid.parquet')
    verify_partition(train,'train')
    verify_partition(valid,'valid')
    if len(train)!=contract['counts']['train'] or len(valid)!=contract['counts']['valid']:
        raise ValueError('El contrato no coincide con los archivos de datos')
    if set(train['id'].astype(str)) & set(valid['id'].astype(str)):
        raise ValueError('Filtración: IDs duplicados entre train y valid')
    result={'contract_id':CONTRACT_ID,'train_n':len(train),'valid_n':len(valid),
            'feature_sets':{'baseline':'G1','secondary':'G1 + G2'},'results':{}}
    predictions=pd.DataFrame({'id':valid['id'].astype('string'),
                               'issue_d':pd.to_datetime(valid['issue_d']),
                               'target':valid['target'].astype(int)})
    num,cat=select_feature_columns(False)
    dummy=DummyClassifier(strategy='most_frequent',random_state=RANDOM_STATE)
    dummy.fit(train[num+cat],train['target'])
    p=dummy.predict_proba(valid[num+cat])[:,1]
    result['results']['dummy']=evaluate_binary(valid['target'].to_numpy(),p)
    predictions['score_dummy']=p
    for name,g2 in (('logreg_g1',False),('logreg_g1_g2',True)):
        num,cat=select_feature_columns(g2)
        model=make_baseline(g2)
        model.fit(train[num+cat],train['target'])
        p=model.predict_proba(valid[num+cat])[:,1]
        result['results'][name]=evaluate_binary(valid['target'].to_numpy(),p)
        predictions['score_'+name]=p
    output_metrics=Path(output_metrics)
    output_metrics.parent.mkdir(parents=True,exist_ok=True)
    output_predictions=Path(output_predictions)
    output_predictions.parent.mkdir(parents=True,exist_ok=True)
    output_metrics.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    predictions.to_csv(output_predictions,index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description='Regresión logística y Dummy sobre validación v4; nunca test')
    parser.add_argument('--data-dir',default=str(PROJECT_ROOT/'data'/'splits'/'lc_36m_v4'))
    parser.add_argument('--metrics',default=str(PROJECT_ROOT/'outputs'/'metrics_validacion_v4.json'))
    parser.add_argument('--predictions',default=str(PROJECT_ROOT/'outputs'/'predictions_valid_v4.csv'))
    args=parser.parse_args()
    result=run(args.data_dir,args.metrics,args.predictions)
    for name,m in result['results'].items():
        print(f'{name}: ROC-AUC={m["roc_auc"]:.4f}, PR-AUC={m["pr_auc"]:.4f}, F1={m["f1"]:.4f}')
    print('Solo resultados VALIDACIÓN; test permanece reservado.')

if __name__=='__main__': main()
