"""Uso: python -m scripts.prepare_data --mode legacy"""
import argparse
from src.config import DEFAULT_RAW, PROJECT_ROOT
from src.data import prepare_parquet_parts


def main():
    parser = argparse.ArgumentParser(description='Preparación streaming con whitelist G1/G2.')
    parser.add_argument('--input', default=str(DEFAULT_RAW))
    parser.add_argument('--mode', choices=('legacy','matured','matured_36'), default='legacy')
    parser.add_argument('--output', default=None)
    parser.add_argument('--chunksize', type=int, default=100_000)
    args = parser.parse_args()
    output = args.output or str(PROJECT_ROOT / 'data' / 'processed' / args.mode)
    result = prepare_parquet_parts(args.input, output, args.mode, args.chunksize)
    print(result)

if __name__ == '__main__': main()
