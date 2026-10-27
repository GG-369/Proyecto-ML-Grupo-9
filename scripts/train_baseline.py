"""Reejecuta el baseline sobre la cohorte preparada. NO usa test final."""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier
from src.evaluate import evaluate_binary
from src.features import select_feature_columns
from src.train import train_and_predict
from src.config import PROJECT_ROOT, RANDOM_STATE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(PROJECT_ROOT/'data'/'processed'/'legacy'))
    parser.add_argument('--train-start', default='2012-01-01')
    parser.add_argument('--train-end', default='2014-01-01', help='exclusivo')
    parser.add_argument('--valid-end', default='2014-07-01', help='exclusivo')
    parser.add_argument('--output', default=str(PROJECT_ROOT/'outputs'/'metrics_validacion_local.json'))
    args=parser.parse_args()
    paths=sorted(Path(args.data_dir).glob('part-*.parquet'))
    if not paths: raise FileNotFoundError('No hay Parquet; ejecutar prepare_data primero.')
    df=pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    start, train_end, valid_end=map(pd.Timestamp, (args.train_start,args.train_end,args.valid_end))
    if not start < train_end < valid_end: raise ValueError('Fechas en orden incorrecto.')
    train=df.loc[df['issue_d'].between(start,train_end,inclusive='left')].copy()
    valid=df.loc[df['issue_d'].between(train_end,valid_end,inclusive='left')].copy()
    if train.empty or valid.empty: raise ValueError('train/valid vacío; revisar rango.')
    num,cat=select_feature_columns(False)
    dummy=DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE).fit(train[num+cat],train['target'])
    results={'data_dir':args.data_dir,'train_rows':len(train),'valid_rows':len(valid),
             'train_window':[str(start.date()),str((train_end-pd.Timedelta(days=1)).date())],
             'valid_window':[str(train_end.date()),str((valid_end-pd.Timedelta(days=1)).date())],
             'results':{}}
    results['results']['dummy']=evaluate_binary(valid['target'].to_numpy(),dummy.predict_proba(valid[num+cat])[:,1])
    for label,plus_g2 in [('logreg_G1',False),('logreg_G1_G2',True)]:
        p=train_and_predict(train, valid, include_g2=plus_g2)
        results['results'][label]=evaluate_binary(valid['target'].to_numpy(),p)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Guardado:',output)
    for label,m in results['results'].items():
        print(label,'ROC-AUC:',round(m['roc_auc'],4),'PR-AUC:',round(m['pr_auc'],4))

if __name__=='__main__':main()
