"""Fair evaluation of the Residual CNN and Transformer models.

Run this script after training at least two models.  By default it compares
``cnn.h5`` and ``transformer4.h5`` on the same independent seen-device test
set (devices 1--30, 400 packets per device).

Outputs are written to ``comparison_results/``:
  * summary.csv and summary.md: paper-ready metric tables;
  * <model>_classification_report.csv: per-device precision/recall/F1;
  * <model>_confusion_matrix_{count,normalised}.png: error analysis;
  * convergence.png: optional training/validation loss and accuracy curves.

The ``--history-*`` arguments accept a Keras CSVLogger file.  They are
optional because old H5 checkpoints do not contain their training history.
"""

from __future__ import print_function

import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Importing these modules registers custom Lambda functions used by old H5
# checkpoints.  Do not remove even though the imported names are not called.
import deep_learning_models  # noqa: F401
import transformer2_models  # noqa: F401
import transformer3_models  # noqa: F401
import transformer4_models  # noqa: F401
from dataset_preparation import ChannelIndSpectrogram, LoadDataset
from transformer_models import PositionEmbedding


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.environ.get('LORA_RFFI_DATASET_DIR',
                             os.path.join(PROJECT_DIR, 'dataset'))
DEFAULT_TEST_FILE = os.path.join(DATASET_DIR, 'Test',
                                 'dataset_seen_devices.h5')


def parse_args():
    parser = argparse.ArgumentParser(description='Compare LoRa RFFI models fairly.')
    parser.add_argument('--test-file', default=DEFAULT_TEST_FILE)
    parser.add_argument('--models', nargs='+',
                        default=['CNN=cnn.h5', 'Transformer-V4=transformer4.h5'],
                        help='One or more NAME=MODEL_PATH entries.')
    parser.add_argument('--output-dir', default=os.path.join(PROJECT_DIR, 'comparison_results'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--history-cnn', default=None,
                        help='Optional CSVLogger file for the CNN.')
    parser.add_argument('--history-transformer', default=None,
                        help='Optional CSVLogger file for the Transformer.')
    return parser.parse_args()


def parse_models(items):
    models = []
    for item in items:
        if '=' not in item:
            raise ValueError('Each --models item must have the form NAME=PATH: %s' % item)
        name, path = item.split('=', 1)
        path = path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)
        if not os.path.isfile(path):
            raise IOError('Model file not found: %s' % path)
        models.append((name, path))
    return models


def load_seen_test_data(path):
    """Load exactly the same independent seen-device test split for all models."""
    iq, labels = LoadDataset().load_iq_samples(
        path, dev_range=range(0, 30), pkt_range=range(0, 400))
    features = ChannelIndSpectrogram().channel_ind_spectrogram(iq)
    return features, labels.astype(int)


def model_flops(model):
    """Estimate single-sample FLOPs. Returns NaN if TF cannot profile an H5 graph."""
    try:
        # TensorFlow 2.x uses a concrete graph for profiling.  The result is
        # an estimate; state this clearly in the dissertation.
        input_shape = [1] + list(model.input_shape[1:])
        concrete = tf.function(lambda x: model(x)).get_concrete_function(
            tf.TensorSpec(input_shape, tf.float32))
        frozen = convert_variables_to_constants_v2(concrete)
        graph_def = frozen.graph.as_graph_def()
        with tf.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name='')
            options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            profile = tf.compat.v1.profiler.profile(graph=graph, cmd='op', options=options)
        return float(profile.total_float_ops) if profile is not None else float('nan')
    except Exception as error:  # Some Lambda layers cannot be frozen on TF 2.10.
        print('Warning: FLOP profiling unavailable for this model: %s' % error)
        return float('nan')


def timed_predict(model, features, batch_size):
    """Measure inference after one warm-up pass to avoid CUDA initialisation time."""
    warmup = min(batch_size, len(features))
    model.predict(features[:warmup], batch_size=batch_size, verbose=0)
    start = time.perf_counter()
    probabilities = model.predict(features, batch_size=batch_size, verbose=0)
    elapsed = time.perf_counter() - start
    return probabilities, elapsed


def save_confusion_matrices(name, labels, predictions, output_dir):
    classes = np.arange(1, 31)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(30))
    normalised = matrix.astype(float) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    for data, suffix, fmt, cmap in [
            (matrix, 'count', 'd', 'Blues'),
            (normalised, 'normalised', '.2f', 'magma')]:
        plt.figure(figsize=(13, 11))
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, cbar=True,
                    xticklabels=classes, yticklabels=classes,
                    annot_kws={'size': 5})
        plt.xlabel('Predicted device ID')
        plt.ylabel('True device ID')
        plt.title('%s confusion matrix (%s)' % (name, suffix))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '%s_confusion_matrix_%s.png' %
                                 (safe_name(name), suffix)), dpi=300)
        plt.close()
    return matrix


def safe_name(name):
    return ''.join(char if char.isalnum() else '_' for char in name).strip('_')


def evaluate_one(name, model_path, features, labels, output_dir, batch_size):
    print('\nEvaluating %s: %s' % (name, model_path))
    model = load_model(model_path, compile=False,
                       custom_objects={'PositionEmbedding': PositionEmbedding})
    probabilities, inference_seconds = timed_predict(model, features, batch_size)
    predictions = probabilities.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, labels=np.arange(30),
                                   target_names=['Device %d' % value for value in range(1, 31)],
                                   output_dict=True, zero_division=0)
    matrix = save_confusion_matrices(name, labels, predictions, output_dir)

    report_path = os.path.join(output_dir, '%s_classification_report.csv' % safe_name(name))
    with open(report_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['device', 'precision', 'recall', 'f1-score', 'support'])
        for device in range(1, 31):
            row = report['Device %d' % device]
            writer.writerow([device, row['precision'], row['recall'], row['f1-score'], row['support']])

    return {
        'Model': name,
        'Accuracy': accuracy,
        'Macro Precision': precision,
        'Macro Recall': recall,
        'Macro F1': f1,
        'Parameters': int(model.count_params()),
        'FLOPs (one sample)': model_flops(model),
        'Inference time (s)': inference_seconds,
        'Inference time (ms/sample)': 1000.0 * inference_seconds / len(features),
        'Confusion errors': int(matrix.sum() - np.trace(matrix)),
    }


def read_history(path):
    """Read Keras CSVLogger output; supports both accuracy and acc column names."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r') as handle:
        return list(csv.DictReader(handle))


def plot_convergence(cnn_history, transformer_history, output_dir):
    histories = [('Residual CNN', cnn_history), ('Transformer', transformer_history)]
    histories = [(name, data) for name, data in histories if data]
    if not histories:
        print('No training-history CSV supplied: convergence plot was skipped.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, history in histories:
        epoch = np.arange(1, len(history) + 1)
        for column, style, label in [('loss', '-', 'train loss'), ('val_loss', '--', 'validation loss')]:
            if column in history[0]:
                axes[0].plot(epoch, [float(row[column]) for row in history], style,
                             label='%s %s' % (name, label))
        acc_col = 'accuracy' if 'accuracy' in history[0] else 'acc'
        val_acc_col = 'val_accuracy' if 'val_accuracy' in history[0] else 'val_acc'
        if acc_col in history[0]:
            axes[1].plot(epoch, [float(row[acc_col]) for row in history], '-',
                         label='%s train accuracy' % name)
        if val_acc_col in history[0]:
            axes[1].plot(epoch, [float(row[val_acc_col]) for row in history], '--',
                         label='%s validation accuracy' % name)
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Convergence: loss')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Convergence: accuracy')
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'convergence.png'), dpi=300)
    plt.close(fig)


def write_summary(rows, output_dir):
    columns = list(rows[0].keys())
    csv_path = os.path.join(output_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = os.path.join(output_dir, 'summary.md')
    with open(markdown_path, 'w') as handle:
        handle.write('| ' + ' | '.join(columns) + ' |\n')
        handle.write('|' + '|'.join(['---'] * len(columns)) + '|\n')
        for row in rows:
            formatted = []
            for column in columns:
                value = row[column]
                if isinstance(value, float) and not np.isnan(value):
                    formatted.append('%.4f' % value)
                elif isinstance(value, float) and np.isnan(value):
                    formatted.append('N/A')
                else:
                    formatted.append(str(value))
            handle.write('| ' + ' | '.join(formatted) + ' |\n')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    models = parse_models(args.models)
    features, labels = load_seen_test_data(args.test_file)
    print('Independent test samples: %d' % len(labels))

    rows = [evaluate_one(name, path, features, labels, args.output_dir, args.batch_size)
            for name, path in models]
    write_summary(rows, args.output_dir)
    plot_convergence(read_history(args.history_cnn),
                     read_history(args.history_transformer), args.output_dir)
    print('\nCompleted. Results saved to: %s' % args.output_dir)


if __name__ == '__main__':
    main()
