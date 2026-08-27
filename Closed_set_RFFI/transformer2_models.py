"""Tuned Vision Transformer (Transformer2) for LoRa RFFI spectrograms."""

import numpy as np

from keras import backend as K
from keras.layers import (Add, Conv2D, Dense, Dropout, GlobalAveragePooling1D,
                          Input, Lambda, LayerNormalization, MultiHeadAttention,
                          Reshape)
from keras.models import Model
from keras.regularizers import l2

from transformer_models import PositionEmbedding


def _encoder_block(x, projection_dim, num_heads, mlp_dim, dropout_rate, index):
    """Pre-normalized Transformer encoder with regularized residual paths."""
    prefix = 'transformer2_%d' % index

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
    x = Dense(mlp_dim,
              activation='gelu',
              kernel_regularizer=l2(1e-4),
              name=prefix + '_mlp_expand')(x)
    x = Dropout(dropout_rate, name=prefix + '_mlp_dropout_1')(x)
    x = Dense(projection_dim,
              kernel_regularizer=l2(1e-4),
              name=prefix + '_mlp_project')(x)
    x = Dropout(dropout_rate, name=prefix + '_mlp_dropout_2')(x)
    return Add(name=prefix + '_mlp_residual')([residual, x])


def transformer2_classification_net(datashape,
                                    num_classes,
                                    patch_size=(4, 4),
                                    projection_dim=96,
                                    num_heads=4,
                                    transformer_layers=4,
                                    mlp_dim=192,
                                    dropout_rate=0.1):
    """Build the second, tuned Transformer experiment.

    Four-by-four patches preserve the short, local time-frequency variations
    that carry radio-frequency fingerprints.  The 96-dimensional encoder is
    larger than the V1 baseline but remains practical on a 6 GB RTX 3060.
    """
    if projection_dim % num_heads:
        raise ValueError('projection_dim must be divisible by num_heads.')

    height, width = np.asarray(datashape[1:-1], dtype=int)
    patch_height, patch_width = patch_size
    num_patches = (height // patch_height) * (width // patch_width)
    if num_patches == 0:
        raise ValueError('patch_size must be smaller than the input spectrogram.')

    inputs = Input(shape=(height, width, 1), name='spectrogram')
    x = Conv2D(projection_dim,
               kernel_size=patch_size,
               strides=patch_size,
               padding='valid',
               kernel_regularizer=l2(1e-4),
               name='patch_projection')(inputs)
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
    x = Dense(512, kernel_regularizer=l2(1e-4), name='embedding_dense')(x)
    x = Lambda(lambda tensor: K.l2_normalize(tensor, axis=1),
               name='feature_layer')(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_regularizer=l2(1e-4),
                    name='classification')(x)
    return Model(inputs=inputs, outputs=outputs, name='rffi_transformer2')


classification_net = transformer2_classification_net
