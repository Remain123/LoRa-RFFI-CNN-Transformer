"""Create separate, unit-labelled figures and a complete model comparison table."""

from __future__ import annotations

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, 'experiments')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'comparison_figures')

MODELS = [
    ('cnn', 'Residual CNN'),
    ('transformer', 'Transformer V1'),
    ('transformer2', 'Transformer V2'),
    ('transformer3', 'Transformer V3'),
    ('transformer4', 'Transformer V4 (revised)'),
]


def load_rows():
    rows = []
    for folder, display_name in MODELS:
        path = os.path.join(EXPERIMENTS_DIR, folder, 'metrics.json')
        if not os.path.isfile(path):
            raise FileNotFoundError('Missing experiment record: %s' % path)
        with open(path, 'r') as handle:
            data = json.load(handle)
        test = data['test_metrics']
        rows.append({
            'Model': display_name,
            'Accuracy (%)': 100.0 * test['Accuracy'],
            'Macro precision (%)': 100.0 * test['Macro Precision'],
            'Macro recall (%)': 100.0 * test['Macro Recall'],
            'Macro F1 (%)': 100.0 * test['Macro F1'],
            'Training time (h)': data['total_training_seconds'] / 3600.0,
            'Epochs completed': data['epochs_completed'],
            'Parameters (count)': test['Parameters'],
            'FLOPs/sample (count)': test['FLOPs (one sample)'],
            'Inference latency (ms/sample)': test['Inference time (ms/sample)'],
            'Confusion errors (samples)': test['Confusion errors'],
            'Best validation accuracy (%)': 100.0 * data['best_validation_accuracy'],
            'Best epoch': data['best_epoch_by_val_accuracy'],
        })
    return rows


def bar_chart(rows, field, filename, ylabel, title, scale=1.0, ylim=None):
    labels = [row['Model'] for row in rows]
    values = [row[field] / scale for row in rows]
    fig, ax = plt.subplots(figsize=(9.2, 5.3), dpi=180)
    bars = ax.bar(labels, values, color='#2878b5', width=0.62)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', rotation=18)
    if ylim is not None:
        ax.set_ylim(*ylim)
    upper = ax.get_ylim()[1]
    for bar, value in zip(bars, values):
        if value >= 100:
            text = f'{value:,.0f}'
        elif value >= 10:
            text = f'{value:.2f}'
        else:
            text = f'{value:.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + upper * 0.015,
                text, ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)


def write_table(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    columns = list(rows[0].keys())
    csv_path = os.path.join(OUTPUT_DIR, 'all_model_comparison.csv')
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(OUTPUT_DIR, 'all_model_comparison.md')
    with open(md_path, 'w') as handle:
        handle.write('| ' + ' | '.join(columns) + ' |\n')
        handle.write('|' + '|'.join(['---'] * len(columns)) + '|\n')
        for row in rows:
            values = []
            for field in columns:
                value = row[field]
                if isinstance(value, float):
                    values.append(f'{value:.4f}')
                else:
                    values.append(str(value))
            handle.write('| ' + ' | '.join(values) + ' |\n')

    display_columns = [
        'Model', 'Accuracy (%)', 'Macro precision (%)', 'Macro recall (%)',
        'Macro F1 (%)', 'Training time (h)', 'Epochs completed',
        'Parameters (count)', 'FLOPs/sample (count)',
        'Inference latency (ms/sample)', 'Confusion errors (samples)',
    ]
    table_text = []
    for row in rows:
        table_text.append([
            row['Model'], f"{row['Accuracy (%)']:.2f}",
            f"{row['Macro precision (%)']:.2f}",
            f"{row['Macro recall (%)']:.2f}", f"{row['Macro F1 (%)']:.2f}",
            f"{row['Training time (h)']:.2f}", str(row['Epochs completed']),
            f"{row['Parameters (count)']:,}", f"{row['FLOPs/sample (count)'] / 1e6:.2f} M",
            f"{row['Inference latency (ms/sample)']:.3f}",
            f"{row['Confusion errors (samples)']:,}",
        ])
    headers = [
        'Model', 'Accuracy\n(%)', 'Macro P\n(%)', 'Macro R\n(%)', 'Macro F1\n(%)',
        'Training\ntime (h)', 'Epochs', 'Parameters\n(count)',
        'FLOPs/sample', 'Inference\n(ms/sample)', 'Errors\n(samples)',
    ]
    fig, ax = plt.subplots(figsize=(20, 4.3), dpi=180)
    ax.axis('off')
    column_widths = [0.13, 0.085, 0.085, 0.085, 0.085, 0.095, 0.075,
                     0.11, 0.10, 0.095, 0.095]
    table = ax.table(cellText=table_text, colLabels=headers, cellLoc='center',
                     colWidths=column_widths, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.75)
    for column in range(len(headers)):
        table[(0, column)].set_facecolor('#2878b5')
        table[(0, column)].get_text().set_color('white')
    ax.set_title('Complete model comparison',
                 fontsize=15, pad=16)
    fig.savefig(os.path.join(OUTPUT_DIR, 'all_model_comparison_table.png'),
                bbox_inches='tight')
    plt.close(fig)


def main():
    rows = load_rows()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_table(rows)

    bar_chart(rows, 'Accuracy (%)', 'accuracy.png', 'Accuracy (%)',
              'Test-set classification accuracy', ylim=(80, 101))
    bar_chart(rows, 'Macro precision (%)', 'macro_precision.png', 'Macro precision (%)',
              'Test-set macro precision', ylim=(80, 101))
    bar_chart(rows, 'Macro recall (%)', 'macro_recall.png', 'Macro recall (%)',
              'Test-set macro recall', ylim=(80, 101))
    bar_chart(rows, 'Macro F1 (%)', 'macro_f1.png', 'Macro F1 (%)',
              'Test-set macro F1 score', ylim=(80, 101))
    bar_chart(rows, 'Training time (h)', 'training_time_hours.png', 'Training time (hours)',
              'Total training time')
    bar_chart(rows, 'Epochs completed', 'epochs_completed.png', 'Completed epochs (count)',
              'Number of training epochs')
    bar_chart(rows, 'Parameters (count)', 'parameters_millions.png', 'Trainable parameters (millions)',
              'Trainable parameter count', scale=1e6)
    bar_chart(rows, 'FLOPs/sample (count)', 'flops_millions.png', 'FLOPs per sample (millions)',
              'Computational complexity', scale=1e6)
    bar_chart(rows, 'Inference latency (ms/sample)', 'inference_latency.png',
              'Inference latency (ms/sample)', 'Inference efficiency')
    bar_chart(rows, 'Confusion errors (samples)', 'confusion_errors.png',
              'Misclassified samples (count)', 'Test-set confusion errors')
    print('Saved figures and tables to: %s' % OUTPUT_DIR)


if __name__ == '__main__':
    main()
