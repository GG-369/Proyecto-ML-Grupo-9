"""Uso: python -m scripts.audit_dataset --input data/raw/accepted_2007_to_2018Q4.csv.gz"""
import argparse
from src.config import DEFAULT_RAW, DEFAULT_AUDIT
from src.data import write_audit


def main():
    parser = argparse.ArgumentParser(description='Audita TODO Lending Club en streaming.')
    parser.add_argument('--input', default=str(DEFAULT_RAW))
    parser.add_argument('--output', default=str(DEFAULT_AUDIT))
    parser.add_argument('--chunksize', type=int, default=100_000)
    args = parser.parse_args()
    result = write_audit(args.input, args.output, args.chunksize)
    print('FILAS TOTALES:', result['counts']['raw_total'])
    for key in ['legacy_final', 'all_final', 'matured_final', 'matured_36_final']:
        print(f'{key}:', result['counts'].get(key, 0))
    print('Auditoría guardada en:', args.output)

if __name__ == '__main__': main()
