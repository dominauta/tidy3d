"""Defines a heat solver compatible structure."""
from __future__ import annotations

import pydantic as pd
from typing import Union

from ....components.base import cached_property
from ....components.structure import Structure
from ....components.types import TYPE_TAG_STR

from .medium import HeatMediumType
from .source import HeatSourceType


class HeatStructure(Structure):
    """A :class:`.Structure` registered with jax."""

    medium: HeatMediumType = pd.Field(
        ...,
        title="Medium",
        description="Medium of the structure, which is heat solver compatible.",
        discriminator=TYPE_TAG_STR,
    )

    source: HeatSourceType = pd.Field(
        None,
        title="Source",
        description="Heat source applied inside the strucutre.",
    )

    @cached_property
    def to_structure(self) -> Structure:

        new_dict = self.dict(exclude={"source", "type"})
        new_dict.update({"medium": self.medium.optic_spec})

        return Structure.parse_obj(new_dict)

HeatStructureType = Union[HeatStructure]
