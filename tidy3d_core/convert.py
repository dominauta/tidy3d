""" Tools to switch between new and old Tidy3D formats """
from typing import Dict, Tuple, List
import numpy as np
import h5py

from tidy3d import Simulation
from tidy3d import Box, Sphere, Cylinder, PolySlab
from tidy3d import Medium  # , DispersiveMedium
from tidy3d import VolumeSource
from tidy3d import GaussianPulse
from tidy3d import FieldMonitor, FieldTimeMonitor, FluxMonitor, FluxTimeMonitor
from tidy3d.components.monitor import ScalarFieldMonitor, AbstractFluxMonitor
from tidy3d.components.monitor import FreqMonitor, TimeMonitor
from .solver import discretize_monitor, SolverDataDict


def old_json_parameters(sim: Simulation) -> Dict:
    """Convert simulation parameters to a dict."""

    cent = sim.center
    size = sim.size
    pml_layers = [{"profile": pml.profile, "Nlayers": pml.num_layers} for pml in sim.pml_layers]

    parameters = {
        "unit_length": "um",
        "unit_frequency": "THz",
        "unit_time": "ps",
        "x_cent": cent[0],
        "y_cent": cent[1],
        "z_cent": cent[2],
        "x_span": size[0],
        "y_span": size[1],
        "z_span": size[2],
        "mesh_step": sim.grid_size,
        "symmetries": sim.symmetry,
        "pml_layers": pml_layers,
        "run_time": sim.run_time * 1e12,
        "courant": sim.courant,
        "shutoff": sim.shutoff,
        "subpixel": sim.subpixel,
    }

    """ TODO: Support nonuniform coordinates """
    # if sim.coords is not None:
    #     parameters.update({"coords": [c.tolist() for c in sim.coords]})

    return parameters


def old_json_structures(sim: Simulation) -> Tuple[List[Dict], List[Dict]]:
    """Convert all Structure objects to a list of text-defined geometries, and all the corresponding
    Medium objects to a list of text-defined materials.
    """

    medium_list = []
    medium_map = sim.medium_map
    for (medium, imed) in medium_map.items():
        """TODO: support custom material names (currently no names at all?)"""
        med = {"name": f"mat_{imed}"}
        if isinstance(medium, Medium):
            """TODO: support diagonal anisotropy in non-dispersive media"""
            med.update(
                {
                    "type": "Medium",
                    "permittivity": [medium.permittivity] * 3,
                    "conductivity": [medium.conductivity] * 3,
                    "poles": [],
                }
            )

        """ TODO: Dispersive mediums need to eventually be defined as pole residue pairs for the
        solver. Seems like this will have to be done on the server side in the revamp. """
        # elif isinstance(medium, DispersiveMedium):
        #     poles = []
        #     for pole in medium.poles:
        #         poles.append([pole[0].real, pole[0].imag, pole[1].real, pole[1].imag])
        #     med.update(
        #         {
        #             "type": "Medium",
        #             "permittivity": [medium.eps_inf] * 3,
        #             "conductivity": [0, 0, 0],
        #             "poles": poles,
        #         }
        #     )

        """ TODO: support PEC. Note that PMC is probably not needed (not supported currently)."""
        # elif isinstance(medium, PEC):
        #     med.update({"type": "PEC"})
        medium_list.append(med)

    struct_list = []
    for istruct, structure in enumerate(sim.structures):
        """TODO: Shouldn't structures also have custom names?"""
        struct = {"name": f"struct_{istruct}", "mat_index": medium_map[structure.medium]}
        geom = structure.geometry
        if isinstance(geom, Box):
            cent, size = geom.center, geom.size
            struct.update(
                {
                    "type": "Box",
                    "x_cent": cent[0],
                    "y_cent": cent[1],
                    "z_cent": cent[2],
                    "x_span": size[0],
                    "y_span": size[1],
                    "z_span": size[2],
                }
            )
            struct_list.append(struct)
        elif isinstance(geom, Sphere):
            struct.update(
                {
                    "type": "Sphere",
                    "x_cent": geom.center[0],
                    "y_cent": geom.center[1],
                    "z_cent": geom.center[2],
                    "radius": geom.radius,
                }
            )
            struct_list.append(struct)
        elif isinstance(geom, Cylinder):
            struct.update(
                {
                    "type": "Cylinder",
                    "x_cent": geom.center[0],
                    "y_cent": geom.center[1],
                    "z_cent": geom.center[2],
                    "axis": ["x", "y", "z"][geom.axis],
                    "radius": geom.radius,
                    "height": geom.length,
                }
            )
            struct_list.append(struct)
        elif isinstance(geom, PolySlab):
            struct.update(
                {
                    "type": "PolySlab",
                    "vertices": geom.vertices,
                    "z_cent": (geom.slab_bounds[0] + geom.slab_bounds[1]) / 2,
                    "z_size": geom.slab_bounds[1] - geom.slab_bounds[0],
                    # "slant_angle": geom.sidewall_angle_rad,
                    # "dilation": geom.dilation,
                }
            )
            struct_list.append(struct)

        """ TODO: Support GdsSlab """
        # if isinstance(geom, GdsSlab):
        #     for ip, poly_slab in enumerate(geom.poly_slabs):
        #         poly = struct.copy()
        #         if poly["name"] is None:
        #             poly["name"] = "struct%04d_poly%04d" % (istruct, ip)
        #         else:
        #             poly["name"] += "_poly%04d" % ip
        #         poly.update(
        #             {
        #                 "type": "PolySlab",
        #                 "vertices": poly_slab.vertices.tolist(),
        #                 "z_cent": float(poly_slab.z_cent),
        #                 "z_size": float(poly_slab.z_size),
        #             }
        #         )
        #         struct_list.append(poly)

    return medium_list, struct_list


def old_json_sources(sim: Simulation) -> List[Dict]:
    """Export all sources in the Simulation."""

    src_list = []
    for name, source in sim.sources.items():
        # Get source_time
        if isinstance(source.source_time, GaussianPulse):
            src_time = {
                "type": "GaussianPulse",
                "frequency": source.source_time.freq0 * 1e-12,
                "fwidth": source.source_time.fwidth * 1e-12,
                "offset": source.source_time.offset,
                "phase": source.source_time.phase,
            }

        if isinstance(source, VolumeSource):
            """TODO: Is polarization the right word if we're talking about J and M?
            Check Lumerical notation."""
            component = "E" if source.polarization[0] == "J" else "H"
            component += source.polarization[1]
            src = {
                "name": name,
                "type": "VolumeSource",
                "source_time": src_time,
                "center": source.center,
                "size": source.size,
                "component": component,
                "amplitude": source.source_time.amplitude,
            }
        """ TODO: Support PointDipole as a subclass of VolumeSource """
        # elif isinstance(source, PointDipole):
        #     src = {
        #         "name": src_data.name,
        #         "type": "PointDipole",
        #         "source_time": src_time,
        #         "center": source.center.tolist(),
        #         "size": source.size.tolist(),
        #         "component": source.component,
        #         "amplitude": float(source.amplitude),
        #         }
        """ TODO: Support PlaneWave """
        # elif isinstance(source, PlaneWave):
        #     src = {
        #         "name": src_data.name,
        #         "type": "PlaneWave",
        #         "source_time": src_time,
        #         "injection_axis": source.injection_axis,
        #         "position": source.position,
        #         "polarization": source.polarization,
        #         "amplitude": float(source.amplitude)
        #         }
        """ TODO: Support GaussianBeam """
        # elif isinstance(source, GaussianBeam):
        #     src = {
        #         "name": src_data.name,
        #         "type": "GaussianBeam",
        #         "source_time": src_time,
        #         "position": source.position,
        #         "normal": source.normal,
        #         "direction": source.direction,
        #         "angle_theta": float(source.angle_theta),
        #         "angle_phi": float(source.angle_phi),
        #         "waist_radius": float(source.waist_radius),
        #         "waist_distance": float(source.waist_distance),
        #         "pol_angle": float(source.pol_angle),
        #         "amplitude": float(source.amplitude)
        #         }
        """ TODO: Support ModeSource """
        # elif isinstance(source, ModeSource):
        #     mode_ind = src_data.mode_ind
        #     if mode_ind is None:
        #         log_and_raise(
        #             f"Mode index of source {src_data.name} not yet set, "
        #             "use Simulation.set_mode().",
        #             RuntimeError
        #         )
        #     target_neff = src_data.target_neff
        #     if target_neff is not None:
        #         target_neff = float(target_neff)
        #     src = {
        #         "name": src_data.name,
        #         "type": "ModeSource",
        #         "source_time": src_time,
        #         "center": source.center.tolist(),
        #         "size": source.size.tolist(),
        #         "direction": source.direction,
        #         "amplitude": float(source.amplitude),
        #         "mode_ind": int(mode_ind),
        #         "target_neff": target_neff,
        #         "Nmodes": int(src_data.Nmodes)
        #         }
        src_list.append(src)

    return src_list


def old_json_monitors(sim: Simulation) -> Dict:
    """Export all monitors in the Simulation.

    TODO: All monitors in the old solver interpolate spatially to the center of the Yee cells by
    default. FreqMonitors can be asked to return the fields at the Yee grid locations instead.
    Going forward we should instead always return the raw data from the solver, at sufficiently many
    locations to be able to do any interpolation (user-side) within the monitor.
    """

    mnt_list = []
    for name, monitor in sim.monitors.items():
        mnt = {
            "name": name,
            "x_cent": monitor.center[0],
            "y_cent": monitor.center[1],
            "z_cent": monitor.center[2],
            "x_span": monitor.size[0],
            "y_span": monitor.size[1],
            "z_span": monitor.size[2],
        }
        """ TODO: Time monitors in the revamp work with a TimeSampler that's a sequence of ints,
        presumably the indexes of the discrete time points at which to record. This is currently 
        not supported by the solver, the most freedom there is to record every N steps. Also from a 
        user perspective, the user will rarely want to specifically give the indexes of the
        time points at which to sample. Much more common would be to specify things in units of
        simulation time. We need some convenience function(s). 

        For that, and more generally, it seems that currently the Simulation doesn't have a
        TimeGrid or somethng like it? This is also needed e.g. to plot the source dependence, etc.
        """
        if isinstance(monitor, FieldTimeMonitor):
            store = []
            if np.any([field[0] == "E" for field in monitor.fields]):
                store.append("E")
            if np.any([field[0] == "H" for field in monitor.fields]):
                store.append("H")
            mnt.update(
                {
                    "type": "TimeMonitor",
                    "t_start": 0,
                    "t_stop": sim.run_time,
                    "t_step": None,
                    "store": store,
                }
            )
        elif isinstance(monitor, FluxTimeMonitor):
            mnt.update(
                {
                    "type": "TimeMonitor",
                    "t_start": 0,
                    "t_stop": sim.run_time,
                    "t_step": None,
                    "store": ["flux"],
                }
            )
        elif isinstance(monitor, FieldMonitor):
            store = []
            if np.any([field[0] == "E" for field in monitor.fields]):
                store.append("E")
            if np.any([field[0] == "H" for field in monitor.fields]):
                store.append("H")
            mnt.update(
                {
                    "type": "FrequencyMonitor",
                    "frequency": [f * 1e-12 for f in monitor.freqs],
                    "store": store,
                    "interpolate": True,
                }
            )
        elif isinstance(monitor, FluxMonitor):
            mnt.update(
                {
                    "type": "FrequencyMonitor",
                    "frequency": [f * 1e-12 for f in monitor.freqs],
                    "store": ["flux"],
                    "interpolate": True,
                }
            )
        """ TODO: support ModeMonitor """
        # elif isinstance(monitor, ModeMonitor):
        #     mnt.update(
        #         {
        #             "type": "ModeMonitor",
        #             "frequency": [f * 1e-12 for f in monitor.freqs],
        #             "Nmodes": mnt_data.Nmodes,
        #             "target_neff": mnt_data.target_neff,
        #         }
        #     )
        #
        """ TODO: Support PermittivityMonitor """
        #

        mnt_list.append(mnt)

    return mnt_list


def export_old_json(sim: Simulation) -> Dict:
    """Write a Simulation in the old Tidy3D json format."""

    sim_dict = {}
    sim_dict["parameters"] = old_json_parameters(sim)
    medium_list, struct_list = old_json_structures(sim)
    sim_dict["materials"] = medium_list
    sim_dict["structures"] = struct_list
    sim_dict["sources"] = old_json_sources(sim)
    sim_dict["monitors"] = old_json_monitors(sim)

    return sim_dict


def load_old_monitor_data(simulation: Simulation, data_file: str) -> SolverDataDict:
    """Load a monitor data file from the old Tidy3D format as a dict that can
    be passed to SimulationData.
    """

    data_dict = {}
    f_handle = h5py.File(data_file, "r")
    for name, monitor in simulation.monitors.items():

        if isinstance(monitor, FreqMonitor):
            sampler_values = np.array(monitor.freqs)
            sampler_label = "f"

        elif isinstance(monitor, TimeMonitor):
            sampler_values = np.array(monitor.times)
            sampler_label = "t"

        if isinstance(monitor, ScalarFieldMonitor):
            x, y, z = discretize_monitor(simulation, monitor)
            values = []
            for field in monitor.fields:
                comp = ["x", "y", "z"].index(field[1])
                field_vals = np.array(f_handle[name][field[0]][comp, :, :, :, :])
                values.append(field_vals)

            num_fields = len(monitor.fields)
            x_expanded = num_fields * [x]
            y_expanded = num_fields * [y]
            z_expanded = num_fields * [z]
            data_dict[name] = {
                "fields": monitor.fields,
                "x": x_expanded,
                "y": y_expanded,
                "z": z_expanded,
                "values": values,
                sampler_label: sampler_values,
            }
        elif isinstance(monitor, AbstractFluxMonitor):
            values = np.array(f_handle[name]["flux"]).ravel()
            data_dict[name] = {"values": values, sampler_label: sampler_values}

        # TODO: This should not be needed.
        data_dict[name]["monitor_name"] = name

    f_handle.close()
    return data_dict