from __future__ import annotations

"""Thin re-export module for MAMA-MIA 3D datamodules.

The 2D TorchIO-based datamodules and augmentation code have been removed.
Use `MamaMIA3DKSpaceDataModule` for all current experiments.
"""

from .mamamia_3d import MamaMIA3DKSpaceDataModule

__all__ = ["MamaMIA3DKSpaceDataModule"]
