"""Compatibility with the retained, corrected temporal-window geometry."""
from src.native_spokes.acquisition import RevisionRadialOperator

VERSION = 'native-paired-post-directions-v1'
REDUCTIONS = (10, 12, 16, 24, 32, 48)


def paired_operators(geometry):
    return [RevisionRadialOperator(geometry['trajectory'][t], geometry['dcf'][t],
                                  geometry['constant_image_gain'][t]) for t in range(2)]
