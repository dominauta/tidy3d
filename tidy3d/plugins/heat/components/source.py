"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC
from typing import Union

import pydantic as pd

from .viz import plot_params_heat_source

from ....components.base import Tidy3dBaseModel, cached_property
from ....components.geometry import GeometryType
from ....components.data.data_array import TimeDataArray  #, ScalarFieldTimeDataArray
from ....components.viz import PlotParams

from ....constants import VOLUMETRIC_HEAT_RATE


class HeatSource(ABC, Tidy3dBaseModel):
    """Abstract heat source."""

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a Source object."""
        return plot_params_heat_source


class UniformHeatSource(HeatSource):
    """Volumetric heat source.

    Example
    -------
    >>> const_func = TemperatureDependenceConstant(value=1)
    """

    rate: Union[float, TimeDataArray] = pd.Field(
        title="Volumetric Heat Rate",
        description=f"Volumetric rate of heating or cooling (if negative) in units of {VOLUMETRIC_HEAT_RATE}.",
        units=VOLUMETRIC_HEAT_RATE,
    )


#class CustomHeatSource(HeatSource):
#    """Spatially dependent volumetric heat source.

#    Example
#    -------
#    >>> const_func = TemperatureDependenceConstant(value=1)
#    """

#    geometry: GeometryType = pd.Field(
#        title="Source Geometry",
#        description="Geometry of the heat source.",
#    )

#    rate: ScalarFieldTimeDataArray = pd.Field(
#        title="Volumetric Heat Rate",
#        description="Spatially dependent volumetric rate of heating or cooling (if negative) in units of {VOLUMETRIC_HEAT_RATE}.",
#        units=VOLUMETRIC_HEAT_RATE,
#    )


HeatSourceType = Union[
    UniformHeatSource,
#    HeatCustomSource,
]