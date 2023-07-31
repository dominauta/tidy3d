"""Defines heat grid specifications"""
from __future__ import annotations

import pydantic as pd
from typing import Union

from ....components.base import Tidy3dBaseModel
from ....constants import MICROMETER


class UniformHeatGrid(Tidy3dBaseModel):

    """Uniform grid.

    Example
    -------
    >>> heat_grid = UniformHeatGrid(dl=0.1)
    """

    dl: pd.PositiveFloat = pd.Field(
        ...,
        title="Grid Size",
        description="Grid size for uniform grid generation.",
        units=MICROMETER,
    )

    min_edges_per_circumference: pd.PositiveFloat = pd.Field(
        15,
        title="Minimum edges per circumference",
        description="Enforced minimum number of mesh segments per circumference of an object.",
    )

    min_edges_per_side: pd.PositiveFloat = pd.Field(
        2,
        title="Minimum edges per side",
        description="Enforced minimum number of mesh segments per any side of an object.",
    )


HeatGridType = Union[UniformHeatGrid]
