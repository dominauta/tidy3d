""" utilities for heat solver plotting """
from ....components.viz import PlotParams

""" Constants """

HEAT_BC_COLOR_TEMPERATURE = "orange"
HEAT_BC_COLOR_FLUX = "green"
HEAT_BC_COLOR_CONVECTION = "brown"

# defaults for different tidy3d objects
plot_params_heat_bc = PlotParams(lw=3)
plot_params_heat_source = PlotParams(alpha=0.4, facecolor="red", edgecolor="red", lw=3, hatch="oo")
