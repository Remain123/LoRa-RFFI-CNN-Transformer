"""Transformer classifier for channel-independent LoRa spectrograms.

The public factory intentionally matches ``classification_net`` in
``deep_learning_models.py``: it accepts the full training array shape and the
number of device classes, and returns a softmax classifier with a
``feature_layer`` embedding.  This makes CNN and Transformer experiments
directly comparable in ``main.py``.
"""

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import (Add, Conv2D, Dense, Dropout, Embedding,
                          GlobalAveragePooling1D, Input, LayerNormalization,
                          Lambda, Layer, MultiHeadAttention, Reshape)
from keras.models import Model


@tf.keras.utils.register_keras_serializable(package='rffi')
class PositionEmbedding(Layer):
    """Add trainable, serializable positional embeddings to patch tokens."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.embedding = Embedding(input_dim=num_patches,
                                   output_dim=projection_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embeddings = self.embedding(positions)
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)
        return inputs + position_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return config


def _transformer_block(x, projection_dim, num_heads, mlp_dim, dropout_rate,
                       block_index):
    """Apply one pre-normalized Transformer encoder block."""
    prefix = 'transformer_%d' % block_index

    skip = x
    x = LayerNormalization(epsilon=1e-6, name=prefix + '_attention_norm')(x)
    x = MultiHeadAttention(num_heads=num_heads,
                           key_dim=projection_dim // num_heads,
                           dropout=dropout_rate,
                           name=prefix + '_attention')(x, x)
    x = Dropout(dropout_rate, name=prefix + '_attention_dropout')(x)
    x = Add(name=prefix + '_attention_residual')([skip, x])

    skip = x
    x = LayerNormalization(epsilon=1e-6, name=prefix + '_mlp_norm')(x)
    x = Dense(mlp_dim, activation='gelu', name=prefix + '_mlp_expand')(x)
    x = Dropout(dropout_rate, name=prefix + '_mlp_dropout_1')(x)
    x = Dense(projection_dim, name=prefix + '_mlp_project')(x)
    x = Dropout(dropout_rate, name=prefix + '_mlp_dropout_2')(x)
    return Add(name=prefix + '_mlp_residual')([skip, x])


def transformer_classification_net(datashape,
                                   num_classes,
                                   patch_size=(8, 8),
                                   projection_dim=64,
                                   num_heads=4,
                                   transformer_layers=4,
                                   mlp_dim=128,
                                   dropout_rate=0.1):
    """Build a compact Vision Transformer for RFFI classification.

    Parameters are deliberately modest for the 102 x 62 spectrograms and an
    RTX 3060 Laptop GPU.  ``patch_size=(8, 8)`` produces 84 tokens, avoiding
    the very high memory cost of attention over individual time-frequency
    pixels.
    """
    if projection_dim % num_heads != 0:
        raise ValueError('projection_dim must be divisible by num_heads.')

    height, width = np.asarray(datashape[1:-1], dtype=int)
    patch_height, patch_width = patch_size
    num_patches = (height // patch_height) * (width // patch_width)
    if num_patches == 0:
        raise ValueError('patch_size must be smaller than the input spectrogram.')

    inputs = Input(shape=(height, width, 1), name='spectrogram')

    # A strided convolution performs patch extraction and linear projection.
    x = Conv2D(projection_dim,
               kernel_size=patch_size,
               strides=patch_size,
               padding='valid',
               name='patch_projection')(inputs)
    x = Reshape((num_patches, projection_dim), name='patch_tokens')(x)

    # Trainable positional encodings retain the location of every patch.
    x = PositionEmbedding(num_patches, projection_dim,
                          name='position_embedding')(x)

    for block_index in range(transformer_layers):
        x = _transformer_block(x, projection_dim, num_heads, mlp_dim,
                               dropout_rate, block_index)

    x = LayerNormalization(epsilon=1e-6, name='encoder_norm')(x)
    x = GlobalAveragePooling1D(name='token_average_pool')(x)
    x = Dropout(dropout_rate, name='classifier_dropout')(x)
    x = Dense(512, name='embedding_dense')(x)
    x = Lambda(lambda tensor: K.l2_normalize(tensor, axis=1),
               name='feature_layer')(x)
    outputs = Dense(num_classes, activation='softmax', name='classification')(x)

    return Model(inputs=inputs, outputs=outputs, name='rffi_transformer')


# Drop-in replacement for deep_learning_models.classification_net.  Change
# only the import in main.py to switch between the CNN and Transformer.
classification_net = transformer_classification_net
