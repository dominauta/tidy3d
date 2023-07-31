"""Web API for heat simulation"""

from __future__ import annotations
from typing import Optional, Callable

from datetime import datetime
import os
import pathlib
import tempfile
import time

import pydantic
import rich

from ...exceptions import WebError
from ...log import log
from ...web.http_management import http
from ...web.s3utils import download_file, upload_file, upload_string
from ...web.simulation_task import Folder, SIMULATION_JSON
from ...web.types import ResourceLifecycle, Submittable
from ...web.webapi import wait_for_connection

from .components.simulation import HeatSimulation
from .components.data import HeatSimulationData

HEATSOLVER_API = "tidy3d/heatsolver/py"
HEATSOLVER_HDF5 = "heat_simulation.hdf5"
HEATSOLVER_LOG = "output/result.log"
HEATSOLVER_RESULT = "output/result.hdf5"

STATUS_INQUIRY_FREQ = 0.5  # frequency of checking on simulation status in seconds


# pylint:disable=too-many-arguments
@wait_for_connection
def run(
    heat_simulation: HeatSimulation,
    task_name: str = "Untitled",
    heat_simulation_name: str = "heat_simulation",
    folder_name: str = "Heat Solver",
    results_file: str = "heat_simulation_data.hdf5",
    verbose: bool = True,
    progress_callback_upload: Callable[[float], None] = None,
    progress_callback_download: Callable[[float], None] = None,
) -> HeatSimulationData:
    """Submits a :class:`.HeatSimulation` to server, starts running, monitors progress, downloads,
    and loads results as a :class:`.HeatSimulationData` object.

    Parameters
    ----------
    heat_simulation : :class:`.HeatSimulation`
        Heat simulation to upload to server.
    task_name : str = "Untitled"
        Name of task.
    heat_simulation_name: str = "heat_simulation"
        The name of the heat simulation to create the in task.
    folder_name : str = "Heat Solver"
        Name of folder to store task on web UI.
    results_file : str = "heat_simulation_data.hdf5"
        Path to download results file (.hdf5).
    verbose : bool = True
        If `True`, will print status, otherwise, will run silently.
    progress_callback_upload : Callable[[float], None] = None
        Optional callback function called when uploading file with ``bytes_in_chunk`` as argument.
    progress_callback_download : Callable[[float], None] = None
        Optional callback function called when downloading file with ``bytes_in_chunk`` as argument.

    Returns
    -------
    :class:`.HeatSimulationData`
        Heat simulation data with the calculated results.
    """

    log_level = "DEBUG" if verbose else "INFO"
    if verbose:
        console = rich.console.Console()

    task = HeatSimulationTask.create(heat_simulation, task_name, heat_simulation_name, folder_name)
    if verbose:
        console.log(
            f"Heat simulation created with task_id='{task.task_id}', solver_id='{task.solver_id}'."
        )
    task.upload(verbose=verbose, progress_callback=progress_callback_upload)
    task.submit()

    # Wait for task to finish
    prev_status = "draft"
    status = task.status
    while status not in ("success", "error", "diverged", "deleted"):
        if status != prev_status:
            log.log(log_level, f"Heat simulation status: {status}")
            if verbose:
                console.log(f"Heat simulation status: {status}")
            prev_status = status
        time.sleep(STATUS_INQUIRY_FREQ)
        status = task.get_info().status

    if status == "error":
        raise WebError("Error runnig heat simulation.")

    log.log(log_level, f"Heat simulation status: {status}")
    if verbose:
        console.log(f"Heat simulation status: {status}")

    if status != "success":
        # Our cache discards None, so the user is able to re-run
        return None

    return task.get_result(
        to_file=results_file, verbose=verbose, progress_callback=progress_callback_download
    )


class HeatSimulationTask(ResourceLifecycle, Submittable, extra=pydantic.Extra.allow):
    """Interface for managing the running of a :class:`.HeatSimulation` task on server."""

    task_id: str = pydantic.Field(
        None,
        title="task_id",
        description="Task ID number, set when the task is created, leave as None.",
        alias="refId",
    )

    solver_id: str = pydantic.Field(
        None,
        title="solver",
        description="Solver ID number, set when the task is created, leave as None.",
        alias="id",
    )

    real_flex_unit: float = pydantic.Field(
        None, title="real FlexCredits", description="Billed FlexCredits.", alias="charge"
    )

    created_at: Optional[datetime] = pydantic.Field(
        title="created_at", description="Time at which this task was created.", alias="createdAt"
    )

    status: str = pydantic.Field(
        None,
        title="status",
        description="Heat simulation task status.",
    )

#    file_type: str = pydantic.Field(
#        None,
#        title="file_type",
#        description="File type used to upload the heat simulation.",
#        alias="fileType",
#    )

    heat_simulation: HeatSimulation = pydantic.Field(
        None,
        title="heat_simulation",
        description="Heat simulation being run by this task.",
    )

    # pylint: disable=arguments-differ
    @classmethod
    def create(
        cls,
        heat_simulation: HeatSimulation,
        task_name: str = "Untitled",
        heat_simulation_name: str = "heat_simulation",
        folder_name: str = "Heat Solver",
    ) -> HeatSimulationTask:
        """Create a new heat simulation task on the server.

        Parameters
        ----------
        heat_simulation: :class".HeatSimulation"
            The object that will be uploaded to server in the submitting phase.
        task_name: str = "Untitled"
            The name of the task.
        heat_simulation_name: str = "heat_simulation"
            The name of the heat simulation to create the in task.
        folder_name: str = "Heat Solver"
            The name of the folder to store the task.

        Returns
        -------
        :class:`HeatSimulationTask`
            :class:`HeatSimulationTask` object containing information about the created task.
        """
        folder = Folder.get(folder_name, create=True)

        heat_simulation.to_simulation().validate_pre_upload(source_required=False)
        resp = http.post(
            HEATSOLVER_API,
            {
                "projectId": folder.folder_id,
                "taskName": task_name,
                "heatSimulationName": heat_simulation_name,
                "fileType": "Hdf5",  # FIXME: still needed?
                #"fileType": "Hdf5" if len(heat_simulation.simulation.custom_datasets) > 0 else "Json",
            },
        )
        log.info(
            "Heat simulation created with task_id='%s', solver_id='%s'.", resp["refId"], resp["id"]
        )
        return HeatSimulationTask(**resp, heat_simulation=heat_simulation)

    # pylint: disable=arguments-differ, too-many-arguments
    @classmethod
    def get(
        cls,
        task_id: str,
        solver_id: str,
        to_file: str = "heat_simulation.hdf5",
        #to_file: str = "heat_simulation.json",
        #sim_file: str = "simulation.json",
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> HeatSimulationTask:
        """Get heat simulation task from the server by id.

        Parameters
        ----------
        task_id: str
            Unique identifier of the task on server.
        solver_id: str
            Unique identifier of the heat simulation in the task.
        to_file: str = "heat_simulation.json"
            File to store the heat simulation downloaded from the task.
        #sim_file: str = "simulation.json"
        #    File to store the simulation downloaded from the task.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        :class:`HeatSimulationTask`
            :class:`HeatSimulationTask` object containing information about the task.
        """
        resp = http.get(f"{HEATSOLVER_API}/{task_id}/{solver_id}")
        task = HeatSimulationTask(**resp)
#        heat_simulation = task.get_heatsolver(to_file, sim_file, verbose, progress_callback)
        heat_simulation = task.get_heatsolver(to_file, verbose, progress_callback)
        return task.copy(update={"heat_simulation": heat_simulation})

    @property
    def heat_simulation_path(self):
        """Return the heat simulation path on the server for this task."""
        return f"heat_simulation/{self.solver_id}/"

    def get_info(self) -> HeatSimulationTask:
        """Get the current state of this task on the server.

        Returns
        -------
        :class:`HeatSimulationTask`
            :class:`HeatSimulationTask` object containing information about the task, without the mode
            solver.
        """
        resp = http.get(f"{HEATSOLVER_API}/{self.task_id}/{self.solver_id}")
        return HeatSimulationTask(**resp, heat_simulation=self.heat_simulation)

    def upload(
        self, verbose: bool = True, progress_callback: Callable[[float], None] = None
    ) -> None:
        """Upload this task's 'heat_simulation' to the server.

        Parameters
        ----------
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while uploading the data.
        """

        # Upload simulation as json for GUI display
        upload_string(
            self.task_id,
            self.heat_simulation.to_simulation()._json_string,  # pylint: disable=protected-access
            SIMULATION_JSON,
            verbose=verbose,
            progress_callback=progress_callback,
        )

        # Upload a single HDF5 file with the full data
        file, file_name = tempfile.mkstemp()
        os.close(file)
        self.heat_simulation.to_hdf5(file_name)

        try:
            upload_file(
                self.solver_id,
                file_name,
                HEATSOLVER_HDF5,
                verbose=verbose,
                progress_callback=progress_callback,
            )
        finally:
            os.unlink(file_name)

    # pylint: disable=arguments-differ
    def submit(self):
        """Start the execution of this task.

        The heat simulation must be uploaded to the server with the :meth:`HeatSimulationTask.upload` method
        before this step.
        """
        http.post(f"{HEATSOLVER_API}/{self.task_id}/{self.solver_id}/run")

    # pylint: disable=arguments-differ
    def delete(self):
        """Delete the heat simulation and its corresponding task from the server."""
        # Delete heat simulation
        http.delete(f"{HEATSOLVER_API}/{self.task_id}/{self.solver_id}")
        # Delete parent task
        http.delete(f"tidy3d/tasks/{self.task_id}")

    def get_heatsolver(
        self,
        to_file: str = "heat_simulation.hdf5",
#        sim_file: str = "simulation.json",
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> HeatSimulation:
        """Get heat simulation associated with this task from the server.

        Parameters
        ----------
        to_file: str = "heat_simulation.hdf5"
            File to store the heat simulation downloaded from the task.
        #sim_file: str = "simulation.json"
        #    File to store the simulation downloaded from the task.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        :class:`HeatSimulation`
            :class:`HeatSimulation` object associated with this task.
        """
        to_hdf5 = pathlib.Path(to_file).with_suffix(".hdf5")
        download_file(
            self.task_id,
            self.heat_simulation_path + HEATSOLVER_HDF5,
            to_file=to_hdf5,
            verbose=verbose,
            progress_callback=progress_callback,
        )
        heat_simulation = HeatSimulation.from_hdf5(to_hdf5)

        return heat_simulation

    def get_result(
        self,
        to_file: str = "heat_simulation_data.hdf5",
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> HeatSimulationData:
        """Get heat simulation results for this task from the server.

        Parameters
        ----------
        to_file: str = "heat_simulation_data.hdf5"
            File to store the heat simulation downloaded from the task.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        :class:`.HeatSimulationData`
            Heat simulation data with the calculated results.
        """
        download_file(
            self.task_id,
            self.heat_simulation_path + HEATSOLVER_RESULT,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )
        data = HeatSimulationData.from_hdf5(to_file)

        return data

    def get_log(
        self,
        to_file: str = "heat_simulation.log",
        verbose: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> pathlib.Path:
        """Get execution log for this task from the server.

        Parameters
        ----------
        to_file: str = "heat_simulation.log"
            File to store the heat simulation downloaded from the task.
        verbose: bool = True
            Whether to display progress bars.
        progress_callback : Callable[[float], None] = None
            Optional callback function called while downloading the data.

        Returns
        -------
        path: pathlib.Path
            Path to saved file.
        """
        return download_file(
            self.task_id,
            self.heat_simulation_path + HEATSOLVER_LOG,
            to_file=to_file,
            verbose=verbose,
            progress_callback=progress_callback,
        )
