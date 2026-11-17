"""Tests con datos sintéticos: no aportan observaciones reales al proyecto."""
import pandas as pd
from scripts.audit_cohort_detail import audit_cohort_detail


def test_cohort_breakdown_and_overlap(tmp_path):
    data = pd.DataFrame({
        'issue_d': ['Jan-2012','Jan-2012','Mar-2014','Oct-2014',
                    'Apr-2015','Apr-2015','Feb-2018','Mar-2014'],
        'term': ['36 months','60 months','36 months','36 months',
                 '36 months','60 months','36 months','36 months'],
        'loan_status': ['Fully Paid','Charged Off','Default','Fully Paid',
                        'Charged Off','Fully Paid','Current','Current'],
    })
    file = tmp_path/'test.csv.gz'
    data.to_csv(file, index=False, compression='gzip')
    result = audit_cohort_detail(file, chunksize=2)
    assert result['raw_rows_scanned'] == 8
    assert result['cohort_totals']['legacy_final']['final'] == 3
    assert result['cohort_totals']['matured_final']['final'] == 5
    assert result['cohort_totals']['matured_36_final']['final'] == 4
    assert result['overlap']['legacy_and_matured'] == 3
    assert result['overlap']['matured_not_legacy'] == 2
    assert result['overlap']['matured_36_not_legacy'] == 2
    splits=result['candidate_36_month_windows']
    assert splits['train_2012_2013']['final'] == 1
    assert splits['valid_2014H1']['final'] == 1
    assert splits['test_reservado_2014H2_2015H1']['final'] == 2
    assert result['cohort_totals']['matured_36_final']['bad_rate_among_final'] == 0.5
    assert result['cohort_by_issue_year']['matured_36_final']['2015']['bad'] == 1
