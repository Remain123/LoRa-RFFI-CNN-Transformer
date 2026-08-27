"""Frozen entry point for the first Transformer baseline.

The baseline implementation remains in ``transformer_models.py`` unchanged.
This module gives that version a stable, explicit name for experiments and
comparison with Transformer2.
"""

from transformer_models import PositionEmbedding, transformer_classification_net


classification_net = transformer_classification_net
