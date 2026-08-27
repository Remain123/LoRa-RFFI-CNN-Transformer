import os
import time
import csv
import json
import glob
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import load_model
from keras.callbacks import (CSVLogger, Callback, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)
from keras.optimizers import Adam, RMSprop

from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram

from deep_learning_models import classification_net as cnn_classification_net
from transformer_models import (PositionEmbedding,
                                classification_net as transformer_classification_net)
from transformer2_models import classification_net as transformer2_classification_net
from transformer3_models import classification_net as transformer3_classification_net
from transformer4_models import classification_net as transformer4_classification_net
from keras.utils import to_categorical
from compare_models import evaluate_one, load_seen_test_data


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, 'experiments')
DATASET_DIR = os.environ.get('LORA_RFFI_DATASET_DIR',
                             os.path.join(PROJECT_DIR, 'dataset'))
TRAIN_FILE = os.path.join(DATASET_DIR, 'Train', 'dataset_training_aug.h5')
SEEN_TEST_FILE = os.path.join(DATASET_DIR, 'Test', 'dataset_seen_devices.h5')


class TrainingTimer(Callback):
    """Persist total training time and the best validation epoch for comparison."""

    def __init__(self, output_path):
        super(TrainingTimer, self).__init__()
        self.output_path = output_path
        self.start_time = None
        self.epoch_seconds = []
        self._epoch_start = None

    def on_train_begin(self, logs=None):
        self.start_time = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.perf_counter() - self._epoch_start)

    def on_train_end(self, logs=None):
        self.total_seconds = time.perf_counter() - self.start_time
        with open(self.output_path, 'w') as handle:
            handle.write('total_training_seconds=%.6f\n' % self.total_seconds)
            handle.write('epochs_completed=%d\n' % len(self.epoch_seconds))
            handle.write('mean_epoch_seconds=%.6f\n' %
                         (sum(self.epoch_seconds) / max(len(self.epoch_seconds), 1)))


def train(file_path_in,
          dev_range=range(0, 30),
          pkt_range=range(0, 1000),
          model_builder=cnn_classification_net,
          optimizer=None,
          batch_size=32,
          checkpoint_path=None,
          history_path=None,
          timing_path=None,
          seed=2026):

    """
    train_feature_extractor trains an RFF extractor using triplet loss.

    INPUT:
        FILE_PATH_IN is the path of training dataset.

        DEV_RANGE is the label range of LoRa devices to train the RFF extractor.

        PKT_RANGE is the range of packets from each LoRa device to train the RFF extractor.

        SNR_RANGE is the SNR range used in data augmentation.

    RETURN:
        MODEL is trained classification neural network.
    """

    # Reuse the same shuffle and AWGN realisation for every model run.
    # This does not make GPU operations perfectly deterministic, but it makes
    # the data pipeline comparable across CNN and Transformer experiments.
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Load preamble IQ samples and labels.
    LoadDatasetObj = LoadDataset()
    data_train, label_train = LoadDatasetObj.load_iq_samples(file_path=file_path_in,
                                                             dev_range=dev_range,
                                                             pkt_range=pkt_range)

    # Shuffle the training data and labels.
    index = np.arange(len(label_train))
    np.random.shuffle(index)
    data_train = data_train[index, :]
    label_train = label_train[index]

    # One-hot encoding
    label_train = label_train - dev_range[0]
    label_one_hot = to_categorical(label_train)

    # Add noise to increase system robustness
    data_train = awgn(data_train, range(20, 80))

    # Convert to channel independent spectrogram
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_train)

    # Learning rate scheduler
    # Classification accuracy is the experiment's selection metric.  The
    # cross-entropy still controls learning-rate reduction because it changes
    # smoothly even when accuracy is temporarily flat.
    early_stop = EarlyStopping('val_accuracy', mode='max', min_delta=1e-4,
                               patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau('val_loss', min_delta=0, factor=0.2, patience=10, verbose=1)
    callbacks = [early_stop, reduce_lr]
    if checkpoint_path:
        callbacks.append(ModelCheckpoint(checkpoint_path,
                                         monitor='val_accuracy', mode='max',
                                         save_best_only=True,
                                         verbose=1))
    if history_path:
        callbacks.append(CSVLogger(history_path))
    timer = TrainingTimer(timing_path) if timing_path else None
    if timer:
        callbacks.append(timer)

    # Specify optimizer and deep learning model
    opt = optimizer or RMSprop(learning_rate=1e-3)
    initial_learning_rate = float(tf.keras.backend.get_value(opt.learning_rate))
    model = model_builder(data.shape, len(np.unique(label_train)))
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])

    # Start training
    history = model.fit(data,
                        label_one_hot,
                        epochs=400,
                        shuffle=True,
                        validation_split=0.10,
                        verbose=1,
                        batch_size=batch_size,
                        callbacks=callbacks)

    return model, history, timer, initial_learning_rate


def _history_value(history, names):
    """Return the first present Keras history sequence (old Keras used acc)."""
    for name in names:
        if name in history.history:
            return history.history[name]
    return []


def save_experiment_record(model_type, model, history, timer, optimizer,
                           batch_size, final_model_path, experiment_dir,
                           initial_learning_rate):
    """Evaluate one completed run and add it to the cross-model registry."""
    test_features, test_labels = load_seen_test_data(SEEN_TEST_FILE)
    metrics = evaluate_one(model_type, final_model_path, test_features, test_labels,
                           experiment_dir, batch_size)

    val_accuracy = _history_value(history, ['val_accuracy', 'val_acc'])
    val_loss = _history_value(history, ['val_loss'])
    best_epoch = int(np.argmax(val_accuracy) + 1) if val_accuracy else None
    record = {
        'model': model_type,
        'model_file': final_model_path,
        'test_file': SEEN_TEST_FILE,
        'input_shape': list(model.input_shape),
        'training_packets_per_device': 1000,
        'test_packets_per_device': 400,
        'num_devices': 30,
        'batch_size': batch_size,
        'optimizer': optimizer.__class__.__name__,
        'initial_learning_rate': initial_learning_rate,
        'random_seed': 2026,
        'epochs_completed': len(history.epoch),
        'best_epoch_by_val_accuracy': best_epoch,
        'best_validation_accuracy': float(max(val_accuracy)) if val_accuracy else None,
        'best_validation_loss': float(min(val_loss)) if val_loss else None,
        'total_training_seconds': float(getattr(timer, 'total_seconds', float('nan'))),
        'mean_epoch_seconds': float(np.mean(timer.epoch_seconds)) if timer and timer.epoch_seconds else None,
        'test_metrics': metrics,
    }
    with open(os.path.join(experiment_dir, 'metrics.json'), 'w') as handle:
        json.dump(record, handle, indent=2, allow_nan=True)
    with open(os.path.join(experiment_dir, 'model_architecture.json'), 'w') as handle:
        handle.write(model.to_json())
    with open(os.path.join(experiment_dir, 'model_summary.txt'), 'w') as handle:
        model.summary(print_fn=lambda line: handle.write(line + '\n'))

    update_global_summary(model_type, metrics, record)
    update_all_convergence_plot()
    print('\nExperiment archive created: %s' % experiment_dir)
    print('All-model comparison table: %s' %
          os.path.join(EXPERIMENTS_DIR, 'all_models_summary.csv'))


def update_global_summary(model_type, metrics, record):
    """Upsert a model row in the shared final-comparison table."""
    summary_path = os.path.join(EXPERIMENTS_DIR, 'all_models_summary.csv')
    fields = ['Model', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1',
              'Parameters', 'FLOPs (one sample)', 'Inference time (s)',
              'Inference time (ms/sample)', 'Confusion errors',
              'Training time (s)', 'Epochs completed', 'Best epoch',
              'Best validation accuracy', 'Batch size', 'Optimizer', 'Learning rate']
    row = dict(metrics)
    row.update({'Training time (s)': record.get('total_training_seconds'),
                'Epochs completed': record.get('epochs_completed'),
                'Best epoch': record.get('best_epoch_by_val_accuracy'),
                'Best validation accuracy': record.get('best_validation_accuracy'),
                'Batch size': record.get('batch_size'),
                'Optimizer': record.get('optimizer'),
                'Learning rate': record.get('initial_learning_rate')})
    old_rows = []
    if os.path.isfile(summary_path):
        with open(summary_path, 'r', newline='') as handle:
            old_rows = [item for item in csv.DictReader(handle)
                        if item.get('Model') != model_type]
    old_rows.append(row)
    with open(summary_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in old_rows:
            writer.writerow({field: item.get(field, '') for field in fields})
    update_global_comparison_figure(summary_path)


def update_global_comparison_figure(summary_path):
    """Create a compact figure comparing all archived quantitative metrics."""
    with open(summary_path, 'r', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    names = [row['Model'] for row in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    def values(column):
        result = []
        for row in rows:
            try:
                result.append(float(row[column]))
            except (TypeError, ValueError):
                result.append(np.nan)
        return result

    metric_group = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
    width = 0.18
    for index, metric in enumerate(metric_group):
        axes[0, 0].bar(x + (index - 1.5) * width, values(metric), width, label=metric)
    axes[0, 0].set(title='Classification metrics', ylabel='Score', ylim=(0, 1.05))
    axes[0, 0].legend(fontsize=7)

    charts = [
        ('Training time (s)', 'Training time', 'Seconds'),
        ('Parameters', 'Trainable parameters', 'Count'),
        ('FLOPs (one sample)', 'Computational complexity', 'FLOPs / sample'),
        ('Inference time (ms/sample)', 'Inference efficiency', 'Milliseconds / sample'),
        ('Confusion errors', 'Test-set confusion errors', 'Samples'),
    ]
    for axis, (column, title, ylabel) in zip(axes.flat[1:], charts):
        axis.bar(x, values(column))
        axis.set(title=title, ylabel=ylabel)
        axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    for axis in axes.flat:
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=25, ha='right')
        axis.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(EXPERIMENTS_DIR, 'all_models_overview.png'), dpi=300)
    plt.close(fig)


def archive_test_evaluation(model_type, model_path, batch_size):
    """Run standalone testing and merge the test results into the same archive.

    This is used when ``run_for = 'test'``.  It works whether or not the
    model was trained by the current version of the script.
    """
    experiment_dir = os.path.join(EXPERIMENTS_DIR, model_type)
    os.makedirs(experiment_dir, exist_ok=True)
    test_features, test_labels = load_seen_test_data(SEEN_TEST_FILE)
    metrics = evaluate_one(model_type, model_path, test_features, test_labels,
                           experiment_dir, batch_size)
    metrics_path = os.path.join(experiment_dir, 'metrics.json')
    record = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, 'r') as handle:
            record = json.load(handle)
    record.update({'model': model_type,
                   'model_file': model_path,
                   'test_file': SEEN_TEST_FILE,
                   'test_metrics': metrics,
                   'last_test_evaluation': datetime.now().isoformat(timespec='seconds')})
    with open(metrics_path, 'w') as handle:
        json.dump(record, handle, indent=2, allow_nan=True)
    update_global_summary(model_type, metrics, record)
    update_all_convergence_plot()
    print('\nTest archive updated: %s' % experiment_dir)
    print('All-model comparison table: %s' %
          os.path.join(EXPERIMENTS_DIR, 'all_models_summary.csv'))


def update_all_convergence_plot():
    """Combine every archived history.csv into one fair convergence figure."""
    history_paths = glob.glob(os.path.join(EXPERIMENTS_DIR, '*', 'history.csv'))
    if not history_paths:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    lines_drawn = 0
    for history_path in sorted(history_paths):
        with open(history_path, 'r') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        model_name = os.path.basename(os.path.dirname(history_path))
        epochs = np.arange(1, len(rows) + 1)
        if 'val_loss' in rows[0]:
            axes[0].plot(epochs, [float(row['val_loss']) for row in rows],
                         label=model_name)
            lines_drawn += 1
        accuracy_key = 'val_accuracy' if 'val_accuracy' in rows[0] else 'val_acc'
        if accuracy_key in rows[0]:
            axes[1].plot(epochs, [float(row[accuracy_key]) for row in rows],
                         label=model_name)
    if lines_drawn:
        axes[0].set(xlabel='Epoch', ylabel='Validation loss',
                    title='Convergence comparison: validation loss')
        axes[1].set(xlabel='Epoch', ylabel='Validation accuracy',
                    title='Convergence comparison: validation accuracy')
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(EXPERIMENTS_DIR, 'all_models_convergence.png'), dpi=300)
    plt.close(fig)


def test(file_path_in,
         clf_path_in,
         dev_range=np.arange(0, 30),
         pkt_range=np.arange(0, 400)):

    """
    test_classification performs a classification task and returns the
    classification accuracy.

    INPUT:
        FILE_PATH_IN is the path of enrollment dataset.

        CLF_PATH_IN is the path of classification dataset.

        DEV_RANGE is the label range of LoRa devices.

        PKT_RANGE is the range of packets from each LoRa device.

    RETURN:
        ACC is the overall classification accuracy.
    """

    # Load preamble IQ samples and labels.
    LoadDatasetObj = LoadDataset()
    data_test, label_test = LoadDatasetObj.load_iq_samples(file_path=file_path_in,
                                                           dev_range=dev_range,
                                                           pkt_range=pkt_range)

    label_test = label_test - dev_range[0]

    # Load neural network
    net_test = load_model(clf_path_in,
                          compile=False,
                          custom_objects={'PositionEmbedding': PositionEmbedding})

    # Convert to channel independent spectrogram
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_test)

    # Make prediction
    pred_prob = net_test.predict(data)
    pred_label = pred_prob.argmax(axis=-1)

    # Plot confusion matrix
    conf_mat = confusion_matrix(label_test, pred_label)
    classes = dev_range - dev_range[0] + 1

    plt.figure()
    sns.heatmap(conf_mat, annot=True,
                fmt='d', cmap='Blues',
                annot_kws={'size': 7},
                cbar=False,
                xticklabels=classes,
                yticklabels=classes)

    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.savefig('confusion_matrix.pdf', bbox_inches='tight')
    plt.show()

    return accuracy_score(label_test, pred_label)


if __name__ == '__main__':

    run_for = 'test'
    # Available models: 'cnn', 'transformer' (V1 baseline), 'transformer2',
    # 'transformer3' (single-scale hybrid) and 'transformer4' (multi-scale hybrid).
    model_type = 'transformer4'
    model_configs = {
        'cnn': (cnn_classification_net, RMSprop(learning_rate=1e-3), 32),
        'transformer': (transformer_classification_net,
                        RMSprop(learning_rate=1e-3), 32),
        'transformer2': (transformer2_classification_net,
                         Adam(learning_rate=3e-4, clipnorm=1.0), 16),
        'transformer3': (transformer3_classification_net,
                         Adam(learning_rate=3e-4, clipnorm=1.0), 16),
        'transformer4': (transformer4_classification_net,
                         Adam(learning_rate=3e-4, clipnorm=1.0), 16),
    }
    if model_type not in model_configs:
        raise ValueError('Unknown model_type: %s' % model_type)
    model_builder, optimizer, batch_size = model_configs[model_type]

    if run_for == 'train':

        file_path = TRAIN_FILE
        experiment_dir = os.path.join(EXPERIMENTS_DIR, model_type)
        os.makedirs(experiment_dir, exist_ok=True)

        clf_net, history, timer, initial_learning_rate = train(
            file_path,
            model_builder=model_builder,
            optimizer=optimizer,
            batch_size=batch_size,
            checkpoint_path=os.path.join(experiment_dir, 'best_model.h5'),
            history_path=os.path.join(experiment_dir, 'history.csv'),
            timing_path=os.path.join(experiment_dir, 'training_time.txt'))

        # Keep the legacy root-level model name for existing test scripts,
        # and also store an immutable copy inside this run's archive.
        clf_net.save(model_type + '.h5')
        final_model_path = os.path.join(experiment_dir, 'final_model.h5')
        clf_net.save(final_model_path)
        save_experiment_record(model_type, clf_net, history, timer, optimizer,
                               batch_size, final_model_path, experiment_dir,
                               initial_learning_rate)

    elif run_for == 'test':

        # Closed-set classifier was trained on devices 1–30, so evaluate it
        # against the corresponding known-device test dataset.
        clf_path = os.path.join(PROJECT_DIR, model_type + '.h5')
        archive_test_evaluation(model_type, clf_path, batch_size)
