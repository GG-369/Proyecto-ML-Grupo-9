"""Figuras G1 reproducibles desde scores guardados de Nicolás; solo validación."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def run(predictions, out):
    frame = pd.read_csv(predictions)
    y = frame['target'].to_numpy()
    if set(y) != {0, 1}:
        raise ValueError('La validación debe contener ambas clases.')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    for col, name, color in [
        ('score_decision_tree_g1', 'Árbol G1', '#2166ac'),
        ('score_random_forest_g1', 'Bosque G1', '#b2182b'),
    ]:
        score = frame[col].to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC {auc(fpr, tpr):.3f})', color=color)
        axes[1].plot(recall, precision, label=name, color=color)
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Azar (0.500)')
    axes[1].axhline(y.mean(), color='gray', linestyle='--', label=f'Prevalencia ({y.mean():.3f})')
    axes[0].set(xlabel='Tasa de falsas alarmas', ylabel='Tasa de detección',
                title='ROC en validación, G1', xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel='Recall', ylabel='Precisión', title='Precisión–recall en validación, G1',
                xlim=(0, 1), ylim=(0, 1))
    for axis in axes:
        axis.legend(fontsize=8, loc='best')
        axis.grid(alpha=.2)
    fig.suptitle('Lending Club, préstamos aceptados de 36 meses (2014H2)', fontsize=11)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', default='outputs/predictions_valid_tree_nicolas.csv')
    parser.add_argument('--out', default='reports/figures/validacion_arboles_g1.png')
    args = parser.parse_args()
    run(args.predictions, args.out)


if __name__ == '__main__':
    main()
