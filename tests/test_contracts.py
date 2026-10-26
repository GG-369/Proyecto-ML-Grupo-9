"""Datos creados EN MEMORIA para tests unitarios, jamás métricas científicas."""
import pandas as pd
import numpy as np
import pytest
from src.config import FINAL_STATUSES
from src.data import audit_csv, cohort_masks, prepare_parquet_parts
from src.features import LEAKAGE_COLUMNS, derive_features, select_feature_columns


def test_cohort_windows_and_censoring():
    df=pd.DataFrame({
        'issue_d':['Jan-2012','Jun-2014','Dec-2014','Jun-2015','Jul-2015','Dec-2013','Jul-2013'],
        'term':[' 36 months',' 60 months',' 36 months',' 36 months',' 36 months',' 60 months',' 60 months'],
        'loan_status':['Fully Paid','Charged Off','Current','Fully Paid','Fully Paid','Charged Off','Charged Off'],
    })
    masks=cohort_masks(df)
    assert masks['legacy_final'].sum()==4
    assert masks['all_final'].sum()==6
    # 60 meses posteriores a junio 2013 no cumplen margen conservador.
    assert masks['matured_final'].sum()==2
    assert masks['matured_36_final'].sum()==2


def test_allowlist_no_leakage():
    for include_g2 in (False,True):
        a,b=select_feature_columns(include_g2)
        assert not (set(a+b)&LEAKAGE_COLUMNS)
        assert 'loan_status' not in a+b
        assert 'recoveries' not in a+b


def test_features_no_future_payment():
    df=pd.DataFrame({'issue_d':['Jan-2012'],'term':['36 months'],'emp_length':['< 1 year'],
                     'earliest_cr_line':['Jan-2000'],'fico_range_low':[660],
                     'fico_range_high':[664],'total_pymnt':[99999]})
    out=derive_features(df)
    assert out['term'].iloc[0]==36
    assert out['emp_length'].iloc[0]==0
    assert out['fico_avg'].iloc[0]==662
    assert 11.9<out['hist_crediticio_anios'].iloc[0]<12.1
    assert 'total_pymnt' not in sum((list(x) for x in [select_feature_columns()[0], select_feature_columns()[1]]),[])


def test_audit_all_rows_without_sampling(tmp_path):
    df=pd.DataFrame({'issue_d':['Jan-2012','Jun-2014','Jan-2016','Jan-2018'],
                     'loan_status':['Fully Paid','Default','Current','Charged Off'],
                     'term':['36 months','60 months','36 months','36 months']})
    file=tmp_path/'mini.csv.gz';df.to_csv(file,index=False,compression='gzip')
    audit=audit_csv(file,chunksize=1)
    assert audit['counts']['raw_total']==4
    assert audit['counts']['legacy_final']==2
    assert audit['counts']['all_final']==3
    assert audit['counts']['matured_final']==1
    assert audit['statuses']['Current']==1
