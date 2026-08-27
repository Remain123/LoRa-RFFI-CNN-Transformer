"""Evaluate the saved Transformer3 model on all original training samples."""

import os

from keras.models import load_model
from sklearn.metrics import accuracy_score

import transformer3_models  # Registers the Lambda dependencies used by the H5 model.
from dataset_preparation import ChannelIndSpectrogram, LoadDataset
from transformer_models import PositionEmbedding


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.environ.get('LORA_RFFI_DATASET_DIR',
                             os.path.join(PROJECT_DIR, 'dataset'))
TRAIN_FILE = os.path.join(DATASET_DIR, 'Train', 'dataset_training_aug.h5')
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'transformer3.h5')


def main():
    data_iq, labels = LoadDataset().load_iq_samples(
        TRAIN_FILE, dev_range=range(0, 30), pkt_range=range(0, 1000))
    data = ChannelIndSpectrogram().channel_ind_spectrogram(data_iq)

    model = load_model(MODEL_FILE, compile=False,
                       custom_objects={'PositionEmbedding': PositionEmbedding})
    predictions = model.predict(data, batch_size=32, verbose=1).argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    print('Training-set samples: %d' % len(labels))
    print('Training-set accuracy: %.4f' % accuracy)


if __name__ == '__main__':
    main()
