"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC

import pydantic as pd

from ....components.types import Union, TYPE_TAG_STR
from ....components.base import Tidy3dBaseModel
from ....components.medium import MediumType3D
from ....components.validators import validate_name_str
from ....constants import DENSITY, SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY


# Liquid class
class HeatMedium(ABC, Tidy3dBaseModel):
    """Abstract heat material class."""

    name: str = pd.Field(None, title="Name", description="Optional unique name for medium.")

    _name_validator = validate_name_str()

    optic_spec: MediumType3D = pd.Field(
        title="Optic Specification",
        description="Specification of medium's optic properties.",
        discriminator=TYPE_TAG_STR,
    )


class FluidMedium(HeatMedium):
    """Fluid medium.

    Example
    -------
    >>> solid = FluidMedium(
    >>>     optic_spec=Medium(permittivity=3),
    >>> )
    """


class SolidMedium(HeatMedium):
    """Solid medium.

    Example
    -------
    >>> solid = SolidMedium(
    >>>     optic_spec=Medium(permittivity=3),
    >>>     density=1,
    >>>     capacity=2,
    >>>     conductivity=3,
    >>> )
    """

    density: pd.PositiveFloat = pd.Field(
        title="Material density",
        description=f"Material density in units of {DENSITY}.",
        units=DENSITY,
    )

    capacity: pd.PositiveFloat = pd.Field(
        title="Specific heat capacity",
        description=f"Material density in units of {SPECIFIC_HEAT_CAPACITY}.",
        units=SPECIFIC_HEAT_CAPACITY,
    )

    conductivity: pd.PositiveFloat = pd.Field(
        title="Thermal conductivity",
        description=f"Thermal conductivity of material in units of {THERMAL_CONDUCTIVITY}.",
        units=THERMAL_CONDUCTIVITY,
    )


HeatMediumType = Union[FluidMedium, SolidMedium]
