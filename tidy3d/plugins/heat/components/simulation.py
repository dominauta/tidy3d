"""Defines heat simulation class"""
from __future__ import annotations

# from abc import ABC, abstractmethod
from typing import Tuple, List, Union

import pydantic as pd
# import numpy as np

from shapely.plotting import plot_line
from shapely import LineString, MultiLineString, GeometryCollection

from .medium import HeatMediumType, FluidMedium
from .structure import HeatStructureType, HeatStructure
from .boundary import HeatBCTemperature, HeatBCFlux, HeatBCConvection
from .boundary import HeatBCPlacementType
from .boundary import HeatBCPlacementStructure, HeatBCPlacementStructureStructure
from .boundary import HeatBCPlacementStructureSimulation, HeatBCPlacementSimulation
from .boundary import HeatBCPlacementMediumMedium
from .grid import HeatGridType
from .viz import HEAT_BC_COLOR_TEMPERATURE, HEAT_BC_COLOR_FLUX, HEAT_BC_COLOR_CONVECTION, plot_params_heat_bc
from .viz import plot_params_heat_bc, plot_params_heat_source

from ....components.base import cached_property
from ....components.types import Ax, Shapely, TYPE_TAG_STR
from ....components.viz import add_ax_if_none, equal_aspect, PlotParams
from ....components.simulation import Simulation
from ....components.structure import Structure
from ....components.geometry import Box
from ....components.medium import MediumType3D, Medium
from ....components.data.data_array import SpatialDataArray

from ....exceptions import SetupError, ValidationError
from ....constants import inf
from ....log import log
# from ....constants import KELVIN, DENSITY, SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY, PERMITTIVITY
# from ....constants import CONDUCTIVITY, HEAT_FLUX

HEAT_BACK_STRUCTURE_STR = "<<<HEAT_BACKGROUND_STRUCTURE>>>"


class HeatSimulation(Simulation):
    """Contains all information about heat simulation.

    Example
    -------
    >>> FIXME
    """

    heat_medium: HeatMediumType = pd.Field(
        FluidMedium(optic_spec=Medium()),
        title="Background Medium",
        description="Background medium of simulation, defaults to vacuum if not specified.",
        discriminator=TYPE_TAG_STR,
    )

    heat_structures: Tuple[HeatStructureType, ...] = pd.Field(
        (),
        title="Structures",
        description="Tuple of structures present in simulation. "
        "Note: Structures defined later in this list override the "
        "simulation material properties in regions of spatial overlap.",
    )

    heat_boundary_conditions: Tuple[HeatBCPlacementType, ...] = pd.Field(
        (),
        title="Boundary Conditions",
        description="List of boundary conditions.",
    )

    heat_grid_spec: HeatGridType = pd.Field(
        title="Grid Specification",
        description="Grid specification for heat simulation.",
    )

    heat_domain: Box = pd.Field(
        None,
        title="Heat Simulation Domain",
        description="Domain in which heat simulation is solved. If ``None`` heat simulation is "
        "solved in the entire domain of the Tidy3D simulation."
    )

    @pd.root_validator(skip_on_failure=True)
    def initialize_medium_and_structures(cls, values):

        if values.get("medium") != Medium():
            log.warning("'HeatSimulation.medium' is being overwritten by a derivative of 'HeatSimulation.heat_medium'")

        if values.get("structures") != ():
            log.warning("'HeatSimulation.structures' is being overwritten by a derivative of 'HeatSimulation.heat_structures'")

        heat_medium = values.get("heat_medium")
        medium = heat_medium.optic_spec

        heat_structures = values.get("heat_structures")
        structures = [s.to_structure for s in heat_structures]

        values.update({"medium": medium, "structures": structures})
        return values

    @pd.validator("heat_boundary_conditions", always=True)
    def names_exist(cls, val, values):
        """Error if boundary conditions point to non-existing structures/media"""
        structures = values.get("heat_structures")
        heat_medium = values.get("heat_medium")
        mediums = {structure.medium for structure in structures}
        mediums.add(heat_medium)
        structures_names = {s.name for s in structures}
        mediums_names = {m.name for m in mediums}

        for bc in val:
            if isinstance(bc, (HeatBCPlacementStructure, HeatBCPlacementStructureSimulation)):
                if bc.structure not in structures_names:
                    raise SetupError(
                        f"Structure {bc.structure} provided in a :class:`{bc.type}` "
                        "is not found among simulation structures."
                    )
            if isinstance(bc, (HeatBCPlacementStructureStructure)):
                for ind in range(2):
                    if bc.structures[ind] and bc.structures[ind] not in structures_names:
                        raise SetupError(
                            f"Structure {bc.structures[ind]} provided in a :class:`{bc.type}` "
                            "is not found among simulation structures."
                        )
            if isinstance(bc, (HeatBCPlacementMediumMedium)):
                for ind in range(2):
                    if bc.mediums[ind] not in mediums_names:
                        raise SetupError(
                            f"Material {bc.mediums[ind]} provided in a :class:`{bc.type}` "
                            "is not found among simulation mediums."
                        )
        return val

    def to_simulation(self) -> Simulation:
        """Returns underlying :class:`.Simulation` object."""

        sim_dict = self.dict(
            exclude={
                "type",
                "heat_medium",
                "heat_structures",
                "heat_boundary_conditions",
                "heat_grid_spec",
                "heat_domain",
            }
        )

        new_medium = self.heat_medium.optic_spec
        new_structures = [s.to_structure for s in self.heat_structures]

        sim_dict.update({"medium": new_medium, "structures": new_structures})
        return Simulation.parse_obj(sim_dict)

    def to_perturbed_mediums_simulation(self, temperature: SpatialDataArray) -> Simulation:
        """Returns underlying :class:`.Simulation` object."""

        optic_sim = self.to_simulation()
        return optic_sim.perturbed_mediums_copy(temperature=temperature)

    @cached_property
    def heat_domain_structure(self) -> Structure:
        """Returns structure representing the domain of the :class:`.HeatSimulation`."""

        # Unlike the FDTD Simulation.background_structure, the current one is also used to provide/
        # information about domain in which heat simulation is solved. Thus, we set its boundaries
        # either to self.heat_domain or, if None, to bounding box of self.simulation
        if self.heat_domain:
            heat_domain_actual = self.heat_domain
        else:
            heat_domain_actual = self.bounding_box

#        fdtd_background = self.background_structure
#        return fdtd_background.updated_copy(geometry=heat_domain_actual, name=HEAT_BACK_STRUCTURE_STR)
        return HeatStructure(geometry=heat_domain_actual, medium=self.heat_medium, name=HEAT_BACK_STRUCTURE_STR)

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
        heat_source_alpha: float = None,
        **patch_kwargs,
    ) -> Ax:
        """Plot each of simulation's components on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        source_alpha : float = None
            Opacity of the sources. If ``None``, uses Tidy3d default.
        monitor_alpha : float = None
            Opacity of the monitors. If ``None``, uses Tidy3d default.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        ax = self.plot_structures(ax=ax, x=x, y=y, z=z)
        ax = self.plot_heat_sources(ax=ax, x=x, y=y, z=z, alpha=heat_source_alpha)
        ax = self.plot_symmetries(ax=ax, x=x, y=y, z=z)
        ax = self.plot_pml(ax=ax, x=x, y=y, z=z)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        ax = self.plot_boundaries(ax=ax, x=x, y=y, z=z)
        ax = self.plot_heat_boundaries(ax=ax, x=x, y=y, z=z)
        return ax

    @equal_aspect
    @add_ax_if_none
    def plot_heat_boundaries(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's boundary conditions on a plane defined by one nonzero x,y,z
        coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # get structure list
        structures = [self.heat_domain_structure]
        structures += list(self.heat_structures)

        # construct slicing plane
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        center = Box.unpop_axis(position, (0, 0), axis=axis)
        size = Box.unpop_axis(0, (inf, inf), axis=axis)
        plane = Box(center=center, size=size)

        # get boundary conditions in the plane
        boundaries = self._construct_heat_boundaries(
            structures=structures,
            plane=plane,
            boundary_conditions=self.heat_boundary_conditions,
        )

        # plot boundary conditions
        for (bc, bdry) in boundaries:
            ax = self._plot_boundary_condition(boundary=bdry, condition=bc, ax=ax)

        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)

        # clean up the axis display
        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        return ax

    def _get_bc_plot_params(self, condition: HeatBCPlacementType) -> PlotParams:
        """Constructs the plot parameters for given boundary conditions."""

        plot_params = plot_params_heat_bc
        bc = condition.bc

        if isinstance(bc, HeatBCTemperature):
            plot_params = plot_params.updated_copy(edgecolor=HEAT_BC_COLOR_TEMPERATURE)
        elif isinstance(bc, HeatBCFlux):
            plot_params = plot_params.updated_copy(edgecolor=HEAT_BC_COLOR_FLUX)
        elif isinstance(bc, HeatBCConvection):
            plot_params = plot_params.updated_copy(edgecolor=HEAT_BC_COLOR_CONVECTION)

        return plot_params

    def _plot_boundary_condition(self, boundary: Shapely, condition: HeatBCPlacementType, ax: Ax) -> Ax:
        """Plot a structure's cross section shape for a given boundary condition."""
        plot_params_bc = self._get_bc_plot_params(condition=condition)
        ax = self.plot_line(line=boundary, plot_params=plot_params_bc, ax=ax)
        return ax

    # FIXME: probably needs revision
    def plot_line(self, line: Shapely, plot_params: PlotParams, ax: Ax) -> Ax:
        """Defines how a line is plotted on a matplotlib axes."""

        if isinstance(line, (MultiLineString, GeometryCollection)):
            lines = line.geoms
        elif isinstance(line, LineString):
            lines = [line]

        for l in lines:
            plot_line(l, ax=ax, add_points=False, color=plot_params.edgecolor, linewidth=plot_params.linewidth)
            # ax.add_artist(patch)
        return ax

    # pylint:disable=too-many-locals
    @staticmethod
    def _construct_heat_boundaries(
        structures: List[HeatStructure],
        plane: Box,
        boundary_conditions: List[HeatBCPlacementType],
    ) -> List[Tuple[HeatBCPlacementType, Shapely]]:
        """Compute list of boundary lines to plot on plane.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            list of structures to filter on the plane.
        plane : :class:`.Box`
            target plane.
        boundary_conditions : List[HeatBCPlacementType]
            list of boundary conditions associated with structures.

        Returns
        -------
        List[Tuple[:class:`.HeatBCPlacementType`, shapely.geometry.base.BaseGeometry]]
            List of boundary lines and boundary conditions on the plane after merging.
        """

        # get structures in the plane and present named structures and media
        shapes = []
        named_structures_present = set()
        named_mediums_present = set()
        for structure in structures:

            # get list of Shapely shapes that intersect at the plane
            shapes_plane = structure.geometry.intersections_2dbox(plane)

            # append each of them and their medium information to the list of shapes
            for shape in shapes_plane:
                shapes.append((structure.name, structure.medium, shape, shape.bounds))

            # also keep track of present named structures and media
            if structure.name:
                named_structures_present.add(structure.name)

            if structure.medium.name:
                named_mediums_present.add(structure.medium.name)

        background_structure_shape = shapes[0][2]

        # construct an inverse mapping structure -> bc for present structures
        struct_to_bc = {}
        for bc in boundary_conditions:
            if isinstance(bc, (HeatBCPlacementStructure, HeatBCPlacementStructureSimulation)) and bc.structure in named_structures_present:
                if bc.structure in struct_to_bc:
                    struct_to_bc[bc.structure] += [bc]
                else:
                    struct_to_bc[bc.structure] = [bc]

            if isinstance(bc, HeatBCPlacementStructureStructure):
                for structure in bc.structures:
                    if structure in named_structures_present:
                        if structure in struct_to_bc:
                            struct_to_bc[structure] += [bc]
                        else:
                            struct_to_bc[structure] = [bc]

            if isinstance(bc, HeatBCPlacementSimulation):
                struct_to_bc[HEAT_BACK_STRUCTURE_STR] = [bc]

        # construct an inverse mapping medium -> bc for present mediums
        med_to_bc = {}
        for bc in boundary_conditions:
            if isinstance(bc, HeatBCPlacementMediumMedium):
                for med in bc.mediums:
                    if med in named_mediums_present:
                        if med in med_to_bc:
                            med_to_bc[med] += [bc]
                        else:
                            med_to_bc[med] = [bc]

        # construct boundaries in 2 passes:

        # 1. forward foop to take care of Simulation, StructureSimulation, Structure, and MediumMediums
        boundaries = []
        background_shapes = []
        for name, medium, shape, bounds in shapes:

            minx, miny, maxx, maxy = bounds

            # intersect existing boundaries (both structure based and medium based)
            for index, (_bc, _name, _bdry, _bounds) in enumerate(boundaries):

                # simulation bc is overriden only by HeatBCPlacementStructureSimulation
                if isinstance(_bc, HeatBCPlacementSimulation):
                    if name not in struct_to_bc:
                        continue
                    if any(not isinstance(bc, HeatBCPlacementStructureSimulation) for bc in struct_to_bc[name]):
                        continue

                _minx, _miny, _maxx, _maxy = _bounds

                # do a bounding box check to see if any intersection to do anything about
                if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                    continue

                # look more closely to see if intersected.
                if _bdry.is_empty or not shape.intersects(_bdry):
                    continue

                diff_shape = _bdry - shape

                boundaries[index] = (_bc, _name, diff_shape, diff_shape.bounds)

            # create new srtucture based boundary

            if name in struct_to_bc:
                for bc in struct_to_bc[name]:

                    if isinstance(bc, HeatBCPlacementStructure):
                        bdry = shape.exterior
                        bdry = bdry.intersection(background_structure_shape)
                        boundaries.append((bc, name, bdry, bdry.bounds))

                    if isinstance(bc, HeatBCPlacementSimulation):
                        boundaries.append((bc, name, shape.exterior, shape.exterior.bounds))

                    if isinstance(bc, HeatBCPlacementStructureSimulation):
                        bdry = background_structure_shape.exterior
                        bdry = bdry.intersection(shape)
                        boundaries.append((bc, name, bdry, bdry.bounds))

            # create new medium based boundary, and cut or merge relevant background shapes

            # loop through background_shapes (note: all background are non-intersecting or merged)
            # this is similar to _filter_structures_plane but only mediums participating in BCs are tracked
            for index, (_medium, _shape, _bounds) in enumerate(background_shapes):

                _minx, _miny, _maxx, _maxy = _bounds

                # do a bounding box check to see if any intersection to do anything about
                if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                    continue

                # look more closely to see if intersected.
                if _shape.is_empty or not shape.intersects(_shape):
                    continue

                diff_shape = _shape - shape

                # different medium, remove intersection from background shape
                if medium != _medium and len(diff_shape.bounds) > 0:
                    background_shapes[index] = (_medium, diff_shape, diff_shape.bounds)

                    # in case when there is a bc between two media
                    # create a new boudnary segment
                    for bc in med_to_bc[_medium.name]:
                        if medium.name in bc.mediums:
                            bdry = shape.exterior.intersection(_shape)
                            bdry = bdry.intersection(background_structure_shape)
                            boundaries.append((bc, name, bdry, bdry.bounds))

                # same medium, add diff shape to this shape and mark background shape for removal
                # note: this only happens if this medium is listed in BCs
                else:
                    shape = shape | diff_shape
                    background_shapes[index] = None

            # after doing this with all background shapes, add this shape to the background
            # but only if this medium is listed in BCs
            if medium.name in med_to_bc:
                background_shapes.append((medium, shape, shape.bounds))

            # remove any existing background shapes that have been marked as 'None'
            background_shapes = [b for b in background_shapes if b is not None]

        # filter out empty geometries
        boundaries = [(bc, bdry) for (bc, name, bdry, _) in boundaries if bdry]

        # 2. backward foop to take care of StructureStructure
        # we do it in this way because we define the boundary between
        # two overlapping structures A and B, where A comes before B, as
        # boundary(B) intersected by A
        # So, in this loop as we go backwards through the structures we:
        # - (1) when come upon B, create boundary(B)
        # - (2) cut away from it by other structures
        # - (3) when come upon A, intersect it with A and mark it as complete,
        #   that is, no more further modifications
        boundaries_reverse = []

        for name, _, shape, bounds in shapes[:0:-1]:

            minx, miny, maxx, maxy = bounds

            # intersect existing boundaries
            for index, (_bc, _name, _bdry, _bounds, _completed) in enumerate(boundaries_reverse):

                if not _completed:

                    _minx, _miny, _maxx, _maxy = _bounds

                    # do a bounding box check to see if any intersection to do anything about
                    if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                        continue

                    # look more closely to see if intersected.
                    if _bdry.is_empty or not shape.intersects(_bdry):
                        continue

                    # event (3) from above
                    if name in _bc.structures:
                        new_bdry = _bdry.intersection(shape)
                        boundaries_reverse[index] = (_bc, _name, new_bdry, new_bdry.bounds, True)

                    # event (2) from above
                    else:
                        new_bdry = _bdry - shape
                        boundaries_reverse[index] = (_bc, _name, new_bdry, new_bdry.bounds, _completed)

            # create new boundary (event (1) from above)
            if name in struct_to_bc:
                for bc in struct_to_bc[name]:
                    if isinstance(bc, HeatBCPlacementStructureStructure):
                        bdry = shape.exterior
                        bdry = bdry.intersection(background_structure_shape)
                        boundaries_reverse.append((bc, name, bdry, bdry.bounds, False))

        # filter and append completed boundaries to main list
        for bc, _, bdry, _, is_completed in boundaries_reverse:
            if bdry and is_completed:
                boundaries.append((bc, bdry))

        return boundaries

    @equal_aspect
    @add_ax_if_none
    def plot_heat_sources(
        self, x: float = None, y: float = None, z: float = None, alpha: float = None, ax: Ax = None
    ) -> Ax:
        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        plot_params = plot_params_heat_source.to_kwargs()
        if alpha:
            plot_params["alpha"] = alpha

        bounds = self.bounds
#        for source in self.heat_sources:
#            ax = source.geometry.plot(x=x, y=y, z=z, ax=ax, sim_bounds=bounds, **plot_params)
        for struct in self.heat_structures:
            if struct.source:
                ax = struct.geometry.plot(x=x, y=y, z=z, ax=ax, sim_bounds=bounds, **plot_params)
        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)
        return ax
