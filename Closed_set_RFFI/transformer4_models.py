"""Residual-context CNN-Transformer (Transformer4) for LoRa RFFI spectrograms."""

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


def _multiscale_stem(inputs, filters=32):
    """Extract local texture plus a residual wider-context representation.

    The proven 3x3 local branch from Transformer3 remains the main feature
    path.  A dilated 3x3 convolution has an effective 5x5 receptive field and
    supplies complementary context through a projected residual.  Unlike the
    first Transformer4 version, raw input is not added back after fusion:
    doing so can dilute learned RFF features with unprocessed spectrogram
    values.
    """
    local = Conv2D(filters, 3, padding='same', activation='gelu',
                   kernel_regularizer=l2(WEIGHT_DECAY),
                   name='stem_local_3x3')(inputs)
    context = Conv2D(filters, 3, padding='same', dilation_rate=2,
                     activation='gelu', kernel_regularizer=l2(WEIGHT_DECAY),
                     name='stem_context_dilated_3x3')(local)
    context = Conv2D(filters, 1, padding='same',
                     kernel_regularizer=l2(WEIGHT_DECAY),
                     name='stem_context_projection')(context)
    x = Add(name='stem_context_residual')([local, context])
    return LayerNormalization(epsilon=1e-6, name='stem_norm')(x)


def _encoder_block(x, projection_dim, num_heads, mlp_dim, dropout_rate, index):
    """Pre-normalised Transformer encoder block."""
    prefix = 'transformer4_%d' % index

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


def transformer4_classification_net(datashape,
                                    num_classes,
                                    patch_size=(4, 4),
                                    projection_dim=96,
                                    num_heads=4,
                                    transformer_layers=4,
                                    mlp_dim=192,
                                    dropout_rate=0.1):
    """Build Transformer4 with residual-context stem and padded 4x4 patches."""
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
    x = _multiscale_stem(inputs)
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
    return Model(inputs=inputs, outputs=outputs, name='rffi_transformer4')


classification_net = transformer4_classification_net
