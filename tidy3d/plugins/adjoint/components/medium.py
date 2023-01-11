"""Defines jax-compatible mediums."""
from __future__ import annotations

from typing import Dict, Tuple, Union

import pydantic as pd
import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from ....components.types import Bound, Literal
from ....components.medium import Medium, AnisotropicMedium, CustomMedium
from ....components.geometry import Geometry
from ....components.data.monitor_data import FieldData
from ....components.data.dataset import PermittivityDataset
from ....components.data.data_array import ScalarFieldDataArray

from .base import JaxObject
from .types import JaxFloat, validate_jax_float
from .data.data_array import JaxDataArray
from .data.dataset import JaxPermittivityDataset

# number of integration points per unit wavelength in material
PTS_PER_WVL_INTEGRATION = 20


@register_pytree_node_class
class JaxMedium(Medium, JaxObject):
    """A :class:`.Medium` registered with jax."""

    permittivity: JaxFloat = pd.Field(
        1.0,
        title="Permittivity",
        description="Relative permittivity of the medium. May be a ``jax`` ``DeviceArray``.",
        jax_field=True,
    )

    _sanitize_permittivity = validate_jax_float("permittivity")

    def to_medium(self) -> Medium:
        """Convert :class:`.JaxMedium` instance to :class:`.Medium`"""
        self_dict = self.dict(exclude={"type"}).copy()
        return Medium.parse_obj(self_dict)

    # pylint: disable =too-many-locals
    def _get_volume_disc(
        self, grad_data: FieldData, sim_bounds: Bound, wvl_mat: float
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Get the coordinates and volume element for the inside of the corresponding structure."""

        # find intersecting volume between structure and simulation
        mnt_bounds = grad_data.monitor.geometry.bounds
        rmin, rmax = Geometry.bounds_intersection(mnt_bounds, sim_bounds)

        # assemble volume coordinates and differential volume element
        d_vol = 1.0
        vol_coords = {}
        for coord_name, min_edge, max_edge in zip("xyz", rmin, rmax):

            size = max_edge - min_edge

            # ignore this dimension if there is no thickness along it
            if size == 0:
                continue

            # update the volume element value
            num_cells_dim = int(size * PTS_PER_WVL_INTEGRATION / wvl_mat) + 1
            d_len = size / num_cells_dim
            d_vol *= d_len

            # construct the interpolation coordinates along this dimension
            coords_interp = np.linspace(min_edge + d_len / 2, max_edge - d_len / 2, num_cells_dim)
            vol_coords[coord_name] = coords_interp

        return vol_coords, d_vol

    # pylint:disable=too-many-arguments
    def field_contribution(
        self,
        field: Literal["Ex", "Ey", "Ez"],
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
    ) -> float:
        """Compute the contribution to the VJP from a given field component."""

        vol_coords, d_vol = self._get_volume_disc(
            grad_data=grad_data_fwd, sim_bounds=sim_bounds, wvl_mat=wvl_mat
        )
        e_fwd = grad_data_fwd.field_components[field]
        e_adj = grad_data_adj.field_components[field]
        e_dotted = (e_fwd * e_adj).real
        integrand = e_dotted.isel(f=0).interp(**vol_coords)
        return d_vol * jnp.sum(integrand.values)

    # pylint:disable=too-many-locals
    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
    ) -> JaxMedium:
        """Returns the gradient of the medium parameters given forward and adjoint field data."""

        # integrate the dot product of each E component over the volume, update vjp for epsilon
        vjp_permittivty = 0.0
        for field in ("Ex", "Ey", "Ez"):
            vjp_permittivty += self.field_contribution(
                field=field,
                grad_data_fwd=grad_data_fwd,
                grad_data_adj=grad_data_adj,
                sim_bounds=sim_bounds,
                wvl_mat=wvl_mat,
            )

        return self.copy(update=dict(permittivity=vjp_permittivty))


@register_pytree_node_class
class JaxAnisotropicMedium(AnisotropicMedium, JaxObject):
    """A :class:`.Medium` registered with jax."""

    xx: JaxMedium = pd.Field(
        ...,
        title="XX Component",
        description="Medium describing the xx-component of the diagonal permittivity tensor.",
        jax_field=True,
    )

    yy: JaxMedium = pd.Field(
        ...,
        title="YY Component",
        description="Medium describing the yy-component of the diagonal permittivity tensor.",
        jax_field=True,
    )

    zz: JaxMedium = pd.Field(
        ...,
        title="ZZ Component",
        description="Medium describing the zz-component of the diagonal permittivity tensor.",
        jax_field=True,
    )

    def to_medium(self) -> AnisotropicMedium:
        """Convert :class:`.JaxMedium` instance to :class:`.Medium`"""
        self_dict = self.dict(exclude={"type", "xx", "yy", "zz"}).copy()
        for component in "xyz":
            field_name = component + component
            jax_medium = self.components[field_name]
            self_dict[field_name] = jax_medium.to_medium()
        return AnisotropicMedium.parse_obj(self_dict)

    @classmethod
    def from_tidy3d(cls, tidy3d_obj: AnisotropicMedium) -> JaxAnisotropicMedium:
        """Convert :class:`.Tidy3dBaseModel` instance to :class:`.JaxObject`."""
        obj_dict = tidy3d_obj.dict(exclude={"type", "xx", "yy", "zz"}).copy()
        for component, tidy3d_medium in tidy3d_obj.components.items():
            obj_dict[component] = JaxMedium.from_tidy3d(tidy3d_medium)
        return cls.parse_obj(obj_dict)

    # pylint:disable=too-many-locals
    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
    ) -> JaxMedium:
        """Returns the gradient of the medium parameters given forward and adjoint field data."""

        # integrate the dot product of each E component over the volume, update vjp for epsilon
        vjp_fields = {}
        for component in "xyz":
            field_name = "E" + component
            component_name = component + component
            jax_medium = self.components[component_name]
            vjp_ii = jax_medium.field_contribution(
                field=field_name,
                grad_data_fwd=grad_data_fwd,
                grad_data_adj=grad_data_adj,
                sim_bounds=sim_bounds,
                wvl_mat=wvl_mat,
            )
            vjp_fields[component_name] = JaxMedium(permittivity=vjp_ii)

        return self.copy(update=vjp_fields)


@register_pytree_node_class
class JaxCustomMedium(CustomMedium, JaxObject):
    """A :class:`.Medium` registered with jax."""

    eps_dataset: JaxPermittivityDataset = pd.Field(
        ...,
        title="Permittivity Dataset",
        description="User-supplied dataset containing complex-valued permittivity "
        "as a function of space. Permittivity distribution over the Yee-grid will be "
        "interpolated based on ``interp_method``.",
        jax_field=True,
    )

    @pd.validator("eps_dataset", always=True)
    def _single_frequency(cls, val):
        """Override of inherited validator."""
        return val

    @pd.validator("eps_dataset", always=True)
    def _eps_inf_greater_no_less_than_one_sigma_positive(cls, val):
        """Override of inherited validator."""
        return val

    def eps_dataset_freq(self, frequency: float) -> PermittivityDataset:
        """Override of inherited validator."""
        as_custom_medium = self.to_medium()
        return as_custom_medium.eps_dataset_freq(frequency=frequency)

    def to_medium(self) -> CustomMedium:
        """Convert :class:`.JaxMedium` instance to :class:`.Medium`"""
        self_dict = self.dict(exclude={"type"}).copy()
        eps_field_components = {}
        for dim in "xyz":
            field_name = f"eps_{dim}{dim}"
            data_array = self_dict["eps_dataset"][field_name]
            values = data_array["values"]
            coords = data_array["coords"]
            scalar_field = ScalarFieldDataArray(values, coords=coords)
            eps_field_components[field_name] = scalar_field
        eps_dataset = PermittivityDataset(**eps_field_components)
        self_dict["eps_dataset"] = eps_dataset
        return CustomMedium.parse_obj(self_dict)

    @classmethod
    def from_tidy3d(cls, tidy3d_obj: CustomMedium) -> JaxCustomMedium:
        """Convert :class:`.Tidy3dBaseModel` instance to :class:`.JaxObject`."""
        obj_dict = tidy3d_obj.dict(exclude={"type", "eps_dataset"}).copy()
        eps_dataset = tidy3d_obj.eps_dataset
        field_components = {}
        for dim in "xyz":
            field_name = f"eps_{dim}{dim}"
            data_array = eps_dataset.field_components[field_name]
            values = data_array.values.tolist()
            coords = {key: np.array(val).tolist() for key, val in data_array.coords.items()}
            field_components[field_name] = JaxDataArray(values=values, coords=coords)
        eps_dataset = JaxPermittivityDataset(**field_components)
        obj_dict["eps_dataset"] = eps_dataset
        return cls.parse_obj(obj_dict)

    # pylint:disable=too-many-locals, unused-argument
    def store_vjp(
        self,
        grad_data_fwd: FieldData,
        grad_data_adj: FieldData,
        sim_bounds: Bound,
        wvl_mat: float,
    ) -> JaxMedium:
        """Returns the gradient of the medium parameters given forward and adjoint field data."""
        vjp_field_components = {}
        for dim in "xyz":

            eps_field_name = f"eps_{dim}{dim}"
            field_name = f"E{dim}"

            orig_data_array = self.eps_dataset.field_components[eps_field_name]
            coords = orig_data_array.coords
            interp_coords = {dim_pt: coords[dim_pt] for dim_pt in "xyz" if len(coords[dim_pt]) > 1}
            isel_coords = {dim_pt: 0 for dim_pt in "xyz" if len(coords[dim_pt]) <= 1}
            e_fwd = grad_data_fwd.field_components[field_name]
            e_adj = grad_data_adj.field_components[field_name]
            e_dotted = (e_fwd * e_adj).isel(f=0, **isel_coords).interp(**interp_coords)

            d_vol = 1.0  # TODO: use actual volume element using grid?

            vjp_shape = tuple(len(coord) for _, coord in coords.items())
            vjp_values = (d_vol * e_dotted.real.values).reshape(vjp_shape)
            vjp_data_array = JaxDataArray(values=vjp_values, coords=coords)
            vjp_field_components[eps_field_name] = vjp_data_array

        vjp_eps_dataset = JaxPermittivityDataset(**vjp_field_components)
        return self.copy(update=dict(eps_dataset=vjp_eps_dataset))


JaxMediumType = Union[JaxMedium, JaxAnisotropicMedium, JaxCustomMedium]

# pylint: disable=unhashable-member
JAX_MEDIUM_MAP = {
    Medium: JaxMedium,
    AnisotropicMedium: JaxAnisotropicMedium,
    CustomMedium: JaxCustomMedium,
}