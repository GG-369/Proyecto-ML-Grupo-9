"""Auditoría exhaustiva: comparación de cohortes, solapamiento y cortes temporales.

Ejecutar desde la raíz del repositorio:
python -m scripts.audit_cohort_detail

No entrena modelos, no crea etiquetas para préstamos activos y no modifica el dataset.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_RAW, PROJECT_ROOT, FINAL_BAD
from src.data import AUDIT_COLUMNS, cohort_masks, parse_dates, parse_term

COHORTS = ('legacy_final', 'matured_36_final', 'matured_final')
WINDOWS_36 = {
    'train_2012_2013': ('2012-01-01', '2014-01-01'),
    'valid_2014H1': ('2014-01-01', '2014-07-01'),
    'test_reservado_2014H2_2015H1': ('2014-07-01', '2015-07-01'),
}


def _update(target, good, bad, term36, term60):
    target['final'] += int(good.sum() + bad.sum())
    target['good'] += int(good.sum())
    target['bad'] += int(bad.sum())
    target['term_36'] += int(term36.sum())
    target['term_60'] += int(term60.sum())


def audit_cohort_detail(path, chunksize=100_000):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'No existe el dataset: {path}')
    # No contar resultados solo entre finales y esconder la fracción sin etiqueta.
    totals = {name: defaultdict(int) for name in COHORTS}
    by_year = {name: defaultdict(lambda: defaultdict(int)) for name in COHORTS}
    window_counts = {name: defaultdict(int) for name in WINDOWS_36}
    overlap = defaultdict(int)
    seen = 0
    for chunk in pd.read_csv(path, usecols=AUDIT_COLUMNS, compression='infer',
                             chunksize=chunksize, low_memory=False):
        seen += len(chunk)
        masks = cohort_masks(chunk)
        issue = parse_dates(chunk['issue_d'])
        term = parse_term(chunk['term'])
        good = chunk['loan_status'].eq('Fully Paid')
        bad = chunk['loan_status'].isin(FINAL_BAD)
        for name in COHORTS:
            selected = masks[name]
            _update(totals[name], selected & good, selected & bad,
                    selected & term.eq(36), selected & term.eq(60))
            for year in issue.loc[selected].dt.year.dropna().astype(int).unique():
                flag = selected & issue.dt.year.eq(year)
                _update(by_year[name][str(year)], flag & good, flag & bad,
                        flag & term.eq(36), flag & term.eq(60))
        overlap['legacy_and_matured'] += int((masks['legacy_final'] & masks['matured_final']).sum())
        overlap['legacy_and_matured_36'] += int((masks['legacy_final'] & masks['matured_36_final']).sum())
        overlap['legacy_not_matured'] += int((masks['legacy_final'] & ~masks['matured_final']).sum())
        overlap['matured_not_legacy'] += int((masks['matured_final'] & ~masks['legacy_final']).sum())
        overlap['matured_36_not_legacy'] += int((masks['matured_36_final'] & ~masks['legacy_final']).sum())
        for window, (start, end) in WINDOWS_36.items():
            flag = masks['matured_36_final'] & issue.ge(pd.Timestamp(start)) & issue.lt(pd.Timestamp(end))
            _update(window_counts[window], flag & good, flag & bad,
                    flag & term.eq(36), flag & term.eq(60))

    def serialize(row):
        result = dict(row)
        result['bad_rate_among_final'] = (row['bad'] / row['final']) if row['final'] else None
        return result

    # La auditoría principal define cuántos préstamos tienen un resultado final;
    # esta auditoría deja visible la distribución en el tiempo y los solapamientos.
    return {
        'source_file': path.name,
        'raw_rows_scanned': seen,
        'cohort_totals': {name: serialize(totals[name]) for name in COHORTS},
        'cohort_by_issue_year': {
            name: {year: serialize(row) for year, row in sorted(by_year[name].items())}
            for name in COHORTS
        },
        'overlap': dict(overlap),
        'candidate_36_month_windows': {name: serialize(window_counts[name]) for name in WINDOWS_36},
        'methodology_warnings': [
            'Auditoría retrospectiva: fechas del snapshot asumidas, no verificadas.',
            'Las cohortes se solapan. No sumar recuentos de cohortes.',
            'Préstamos aún activos quedan sin etiqueta y no se convierten en pagados.',
            'Cohorte 36 meses tiene alcance exclusivamente para préstamos de ese plazo.',
            'Los splits por issue_d NO prueban que los resultados fueran conocidos al inicio de validación.',
            'El conjunto test_reservado NO debe usarse para seleccionar modelos o hiperparámetros.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='Auditoría por cohorte, año y ventanas de emisión')
    parser.add_argument('--input', default=str(DEFAULT_RAW))
    parser.add_argument('--output', default=str(PROJECT_ROOT/'outputs'/'audit_cohort_detail.json'))
    parser.add_argument('--chunksize', default=100_000, type=int)
    args = parser.parse_args()
    result = audit_cohort_detail(args.input, args.chunksize)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Auditoría detallada escrita en:', output)
    print('Préstamos auditados:', result['raw_rows_scanned'])
    for name, v in result['cohort_totals'].items():
        print(name, 'final=', v['final'], 'bad=', v['bad'], 'bad_rate=',
              None if v['bad_rate_among_final'] is None else round(v['bad_rate_among_final'], 4))
    print('Overlap:', result['overlap'])
    print('Ventanas 36 meses:', {n: x['final'] for n, x in result['candidate_36_month_windows'].items()})


if __name__ == '__main__':
    main()
