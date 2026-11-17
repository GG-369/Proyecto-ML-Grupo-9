"""Features explícitas disponibles en origen; nunca inferir automáticamente."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Contrato DEL BASELINE: exactamente las 18 variables del notebook previo.
G1_NUM = [
    'loan_amnt', 'term', 'emp_length', 'annual_inc', 'dti',
    'delinq_2yrs', 'fico_avg', 'inq_last_6mths', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc', 'hist_crediticio_anios',
]
G1_CAT = ['home_ownership', 'verification_status', 'purpose', 'addr_state']
G2_NUM = ['int_rate', 'installment']
G2_CAT = ['grade']

# Columnas reales necesarias antes de ingeniería de features.
RAW_G1_NUM = [c for c in G1_NUM if c not in {'fico_avg', 'hist_crediticio_anios', 'term', 'emp_length'}]
RAW_NEEDED = sorted(set(RAW_G1_NUM + G1_CAT + G2_NUM + G2_CAT + [
    'id', 'issue_d', 'loan_status', 'term', 'emp_length',
    'fico_range_low', 'fico_range_high', 'earliest_cr_line'
]))

# Denegación explícita de variables de desempeño posterior.
LEAKAGE_COLUMNS = frozenset({
    'loan_status', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
    'total_rec_int', 'total_rec_late_fee', 'recoveries',
    'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
    'next_pymnt_d', 'last_credit_pull_d', 'out_prncp', 'out_prncp_inv',
    'last_fico_range_high', 'last_fico_range_low', 'pymnt_plan',
    'hardship_flag', 'debt_settlement_flag',
})


def derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones deterministas por fila; no aprenden del dataset entero."""
    out = frame.copy()
    out['issue_d'] = pd.to_datetime(out['issue_d'], format='%b-%Y', errors='coerce') if not pd.api.types.is_datetime64_any_dtype(out['issue_d']) else out['issue_d']
    for c in RAW_G1_NUM + G2_NUM + ['fico_range_low', 'fico_range_high']:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    out['term'] = pd.to_numeric(out['term'].astype('string').str.extract(r'(\d+)', expand=False), errors='coerce')
    out['emp_length'] = pd.to_numeric(
        out['emp_length'].astype('string').str.replace('< 1 year', '0 years', regex=False)
        .str.extract(r'(\d+)', expand=False), errors='coerce'
    )
    cr = pd.to_datetime(out['earliest_cr_line'], format='%b-%Y', errors='coerce')
    out['fico_avg'] = (out['fico_range_low'] + out['fico_range_high']) / 2
    out['hist_crediticio_anios'] = (out['issue_d'] - cr).dt.days / 365.25
    return out


def select_feature_columns(include_g2=False):
    numeric = G1_NUM + (G2_NUM if include_g2 else [])
    categorical = G1_CAT + (G2_CAT if include_g2 else [])
    selected = numeric + categorical
    if len(selected) != len(set(selected)) or set(selected) & LEAKAGE_COLUMNS:
        raise ValueError('Contrato de features inválido: duplicados o leakage.')
    return numeric, categorical


def make_preprocessor(include_g2=False):
    num, cat = select_feature_columns(include_g2)
    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), num),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), cat),
    ])
