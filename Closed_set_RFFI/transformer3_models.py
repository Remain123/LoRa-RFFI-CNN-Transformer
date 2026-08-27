"""Hybrid CNN-Transformer (Transformer3) for LoRa RFFI spectrograms.

The convolutional stem preserves the local time-frequency patterns that are
important for radio-frequency fingerprints.  A Transformer encoder then models
relations between the resulting local regions across the whole spectrogram.
"""

import numpy as np

from keras import backend as K
from keras.layers import (Add, Conv2D, Dense, Dropout,
                          GlobalAveragePooling1D, Input, Lambda,
                          LayerNormalization, MultiHeadAttention, Reshape,
                          ZeroPadding2D)
from keras.models import Model
from keras.regularizers import l2

from transformer_models import PositionEmbedding


WEIGHT_DECAY = 1e-4


def _conv_stem(inputs, filters=32):
    """Apply a small, stable local feature extractor before tokenisation.

    The first Transformer3 experiment used a residual BatchNorm stem.  Its
    moving statistics were sensitive to this relatively small, augmented
    dataset and the model collapsed to one class.  This single convolution
    retains the desired local inductive bias while leaving the proven
    Transformer2 encoder and its optimisation behaviour essentially intact.
    """
    x = Conv2D(filters, 3, padding='same', activation='gelu',
               kernel_regularizer=l2(WEIGHT_DECAY),
               name='stem_local_conv')(inputs)
    return LayerNormalization(epsilon=1e-6, name='stem_local_norm')(x)


def _encoder_block(x, projection_dim, num_heads, mlp_dim, dropout_rate, index):
    """Pre-normalised Transformer encoder block."""
    prefix = 'transformer3_%d' % index

    residual = x
    x = LayerNormalization(epsilon=1e-6, name=prefix + '_attention_norm')(x)
    x = MultiHeadAttention(num_heads=num_heads,
                           key_dim=projection_dim // num_heads,
                           dropout=dropout_rate,
                           name=prefix + '_attention')(x, x)
    x = Dropout(dropout_rate, name=prefix + '_attention_dropout')(x)
    x = Add(name=prefix + '_attention_residual')([residual, x])

    residual = x
    x = LayerNormalization(epsilon=1e-6, name=prefix + '_mlp_norm')(x)
    x = Dense(mlp_dim, activation='gelu', kernel_regularizer=l2(WEIGHT_DECAY),
              name=prefix + '_mlp_expand')(x)
    x = Dropout(dropout_rate, name=prefix + '_mlp_dropout_1')(x)
    x = Dense(projection_dim, kernel_regularizer=l2(WEIGHT_DECAY),
              name=prefix + '_mlp_project')(x)
    x = Dropout(dropout_rate, name=prefix + '_mlp_dropout_2')(x)
    return Add(name=prefix + '_mlp_residual')([residual, x])


def transformer3_classification_net(datashape,
                                    num_classes,
                                    patch_size=(4, 4),
                                    projection_dim=96,
                                    num_heads=4,
                                    transformer_layers=4,
                                    mlp_dim=192,
                                    dropout_rate=0.1):
    """Build a padded 4x4-patch CNN-Transformer classifier.

    The 102x62 input is padded to 104x64, so all original samples contribute
    to a token.  This yields 26x16=416 tokens without moving to costly 2x2
    patches.
    """
    if projection_dim % num_heads:
        raise ValueError('projection_dim must be divisible by num_heads.')

    height, width = np.asarray(datashape[1:-1], dtype=int)
    patch_height, patch_width = patch_size
    pad_height = (-height) % patch_height
    pad_width = (-width) % patch_width
    padded_height = height + pad_height
    padded_width = width + pad_width
    num_patches = (padded_height // patch_height) * (padded_width // patch_width)

    inputs = Input(shape=(height, width, 1), name='spectrogram')
    x = _conv_stem(inputs)
    if pad_height or pad_width:
        x = ZeroPadding2D(padding=((0, pad_height), (0, pad_width)),
                          name='input_padding')(x)

    x = Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size,
               padding='valid', kernel_regularizer=l2(WEIGHT_DECAY),
               name='patch_projection')(x)
    x = Reshape((num_patches, projection_dim), name='patch_tokens')(x)
    x = PositionEmbedding(num_patches, projection_dim,
                          name='position_embedding')(x)
    x = Dropout(dropout_rate, name='embedding_dropout')(x)

    for index in range(transformer_layers):
        x = _encoder_block(x, projection_dim, num_heads, mlp_dim,
                           dropout_rate, index)

    x = LayerNormalization(epsilon=1e-6, name='encoder_norm')(x)
    x = GlobalAveragePooling1D(name='token_average_pool')(x)
    x = Dropout(dropout_rate, name='classifier_dropout')(x)
    x = Dense(512, kernel_regularizer=l2(WEIGHT_DECAY), name='embedding_dense')(x)
    x = Lambda(lambda tensor: K.l2_normalize(tensor, axis=1),
               name='feature_layer')(x)
    outputs = Dense(num_classes, activation='softmax',
                    kernel_regularizer=l2(WEIGHT_DECAY),
                    name='classification')(x)
    return Model(inputs=inputs, outputs=outputs, name='rffi_transformer3')


classification_net = transformer3_classification_net
