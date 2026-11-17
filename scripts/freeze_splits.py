"""Prepara el contrato común, sin exportar el holdout final.

python -m scripts.prepare_data --mode matured_36
python -m scripts.freeze_splits
"""
import argparse
from src.config import PROJECT_ROOT
from src.splits import freeze_from_parts


def main():
    parser=argparse.ArgumentParser(description='Congelar train y valid del contrato 36m v4; reservar test')
    parser.add_argument('--data-dir', default=str(PROJECT_ROOT/'data'/'processed'/'matured_36'))
    parser.add_argument('--audit', default=str(PROJECT_ROOT/'outputs'/'audit_cohort_detail.json'))
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT/'data'/'splits'/'lc_36m_v4'))
    args=parser.parse_args()
    m=freeze_from_parts(args.data_dir,args.audit,args.output_dir)
    print('Contrato:',m['contract_id'])
    print('Train:',m['counts']['train'],'Valid:',m['counts']['valid'])
    print('Test reservado:',m['counts']['test'],'(NO exportado)')

if __name__=='__main__': main()
