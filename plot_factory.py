# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains a factory for making plots.
"""

from typing import Optional, Tuple
from nptyping import Complex128, Float, Int, NDArray, Shape

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_autocorrelation_plot(
    series: NDArray[Shape["*"], Float], linewidth: float = 5.0, markersize: float = 10.0,
    n_lag: Optional[int] = None
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    # pylint: disable=too-many-locals
    """
    Make a plot of the autocorrelation function (ACF) of a time series.

    Parameters
    ----------
    series : numpy.ndarray, (n_timestep,)
        The series to use.
    linewidth : float, default=5.0
        The width of the ACF plot line.
    markersize : float, default=10.0
        The size of each ACF marker.
    n_lag : optional of int, default=None
        The number of lags to use.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    acf = statsmodels.tsa.stattools.acf(x=series, nlags=n_lag)

    error = 1 / np.sqrt(len(series))

    (fig, axs) = plt.subplots()

    (markers, stems, _) = axs.stem(acf)
    axs.plot([0, len(acf) - 1], [-error, -error], linestyle="dashed", color="black")
    axs.plot([0, len(acf) - 1], [error, error], linestyle="dashed", color="black")

    axs.set_xlabel("Time Lag")
    axs.set_ylabel("Correlation")

    plt.setp(markers, "markersize", markersize)
    plt.setp(stems, "linewidth", linewidth)

    return (fig, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_correlation_heatmap(
    data: NDArray[Shape["*,*"], Float], labels: NDArray[Shape["*"], Int], aspect: str = "equal",
    boundary_color: str = "red", cmap: str = "viridis"
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.pyplot.colorbar]:
    # pylint: disable=too-many-locals
    """
    Make a heatmap of the correlations between loads.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_load)
        The load data to plot.
    labels : numpy.ndarray, (n_load,)
        The phase labels of the loads.
    aspect : str, default="equal", ["auto", "equal"]
        The aspect mode to use.
    boundary_color : str, default="red"
        The color of the phase boundary lines.
    cmap : str, default="viridis"
        The color map to use.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    cbar : matplotlib.pyplot.colorbar
        The `colorbar` object in the figure.
    """
    #***********************************************************************************************
    # Create the dotted boundaries to seperated the loads by phase more distinctly.
    # We want to place the phase labels inbetween the boundaries.
    #***********************************************************************************************
    # Boundaries to show the phases more distinctly using horizontal and vertical lines.
    unique_labels = np.unique(labels)
    phase_counts = np.bincount(labels)[unique_labels]
    boundaries = np.cumsum(phase_counts)

    # Tick positions for the phases in the graph axis. This allows us to position the labels "A",
    # "B", etc. inbetween each boundary lines.
    tick_positions = []
    temp = [0] + list(boundaries)

    for i in range(1, len(temp)):
        distance = temp[i] - temp[i - 1]
        position = temp[i - 1] + (distance / 2)

        tick_positions.append(position)

    tick_labels = np.array(["A", "B", "C", "AB", "AC", "BC", "ABC"])[unique_labels]

    #***********************************************************************************************
    # Make the heatmap and add a color bar on the side to show the correlation value/color
    # relationship.
    #***********************************************************************************************
    # Sort the loads by phase so we can better see how the phases follow the correlation structure
    # of the voltage measurements.
    sort_indices = np.argsort(labels)

    corr = np.corrcoef(data[:, sort_indices], rowvar=False)

    (figure, axs) = plt.subplots()
    axs.imshow(corr, aspect=aspect, cmap=cmap)

    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=np.min(corr), vmax=np.max(corr)), cmap=cmap
    )

    cbar = axs.figure.colorbar(mappable=mappable, ax=axs)
    cbar.ax.set_ylabel("Coefficient")

    #***********************************************************************************************
    # Label the plot's axis' and plot the boundary lines.
    #***********************************************************************************************
    axs.set_xlabel("Load")
    axs.set_ylabel("Load")
    axs.set_xticks(tick_positions)
    axs.set_yticks(tick_positions)
    axs.set_xticklabels(tick_labels)
    axs.set_yticklabels(tick_labels)

    for boundary in boundaries[:-1]:
        axs.axhline(y=boundary, color=boundary_color, linestyle="dashed", linewidth=5)
        axs.axvline(x=boundary, color=boundary_color, linestyle="dashed", linewidth=5)

    return (figure, axs, cbar)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_periodogram_filter_plot(
    filter_dft: NDArray[Shape["*"], Complex128], frequencies: NDArray[Shape["*"], Float],
    linewidth: float = 5.0, cutoff: Optional[float] = None
)-> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a periodogram plot of a filter.

    Parameters
    ----------
    filter_dft : numpy.ndarray of complex, (n_timestep,)
        The filter in the frequency domain.
    frequencies : numpy.ndarray of float, (n_timestep,)
        The the values for the frequency axis.
    linewidth : float, default=1.0
        The width of the periodogram plot line.
    cutoff : optional of float, default=None
        The cutoff frequency of the filter, if it exists. A dashed red vertical line will be drawn
        at the cutoff frequency. A value of `None` will not draw the vertical line.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    magnitudes = np.abs(filter_dft)

    (fig, axs) = plt.subplots()

    axs.plot(np.fft.fftshift(frequencies), np.fft.fftshift(magnitudes), linewidth=linewidth)

    if cutoff is not None:
        axs.axvline(x=cutoff, color="red", linestyle="dashed", linewidth=linewidth)
        # pylint: disable=invalid-unary-operand-type
        axs.axvline(x=-cutoff, color="red", linestyle="dashed", linewidth=linewidth)

    axs.set_xlabel(r"$\omega$ (Cycles/Sample)")
    axs.set_ylabel(r"$|H(\omega)|$")

    return (fig, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_scatter_dr_plot(
    data: NDArray[Shape["*,*"], Float], labels: NDArray[Shape["*"], Int],
    dot_size: Optional[int] = 20
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    # pylint: disable=too-many-locals
    """
    Make a 2-D/3-D scatter plot of the first two/three components of some dimensionality reduction
    technique.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, 2 or 3)
        The load data components to plot.
    labels : numpy.ndarray, (n_load,)
        The phase labels of the loads.
    dot_size : int
        The size of the dots in the scatter plot.
    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    n_dim = 3 if data.shape[1] > 3 else data.shape[1]
    if n_dim not in [2, 3]:
        raise ValueError(f"The data should contain either 2 or 3 components, not {n_dim}")

    unique_labels = np.unique(labels)
    phase_indices = [np.where(labels == i)[0] for i in unique_labels]

    plot_labels = np.array(["A", "B", "C", "AB", "AC", "BC", "ABC"])[unique_labels]
    colors = np.array(
        ["red", "blue", "green", "purple", "orange", "turquoise", "black"]
    )[unique_labels]

    fig = plt.figure()
    axs = fig.add_subplot(111, projection="3d" if (n_dim == 3) else None)

    for (color, indices, plot_label) in zip(colors, phase_indices, plot_labels):
        columns_list = [data[indices, i] for i in range(n_dim)]

        axs.scatter(*columns_list, edgecolors="black", color=color, label=plot_label, s=dot_size)

    axs.legend()

    # Set the axis limits to be squared and shift the image to include all points.
    min_points = [np.min(data[:, i]) for i in range(n_dim)]
    max_points = [np.max(data[:, i]) for i in range(n_dim)]

    distances = [max_points[i] - min_points[i] for i in range(n_dim)]
    box_width = np.max(distances)
    edge_width = box_width * 0.05

    axs.set_xlabel("Component 1")
    axs.set_ylabel("Component 2")
    axs.set_xlim([min_points[0] - edge_width, min_points[0] + box_width + edge_width])
    axs.set_ylim([min_points[1] - edge_width, min_points[1] + box_width + edge_width])

    if n_dim == 3:
        axs.set_zlabel("Component 3")
        axs.set_zlim([min_points[2] - edge_width, min_points[2] + box_width + edge_width])

    axs.set_aspect("equal")
    axs.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    return (fig, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_spectrogram_plot(
    spectrogram: NDArray[Shape["*,*"], Complex128], cutoff: Optional[float] = None
)-> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a spectrogram plot from a matrix of windows of a time series.

    Parameters
    ----------
    spectrogram : numpy.ndarray of numpy.complex128, (width, n_window)
        The spectrogram that will be plotted.
    cutoff : optional of float, default=None
        The cutoff frequency. A dashed horizontal line will be drawn at the cutoff frequency.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    #***********************************************************************************************
    # Get the magnitudes from the spectrogram in dB.
    #***********************************************************************************************
    # Add a small number (1e-6) to prevent log of zero.
    magnitudes_db = 10 * np.log10(np.abs(spectrogram) + 1e-6)
    magnitudes_db = np.fft.fftshift(magnitudes_db, axes=0)

    #***********************************************************************************************
    # Plot the periodogram of each window to get the spectrogram.
    #***********************************************************************************************
    (fig, axs) = plt.subplots()

    fig.tight_layout()

    image = axs.imshow(
        magnitudes_db, cmap="seismic", aspect="auto", interpolation="none",
        extent=[1, spectrogram.shape[1], 0.5, -0.5]
    )

    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=np.min(magnitudes_db), vmax=np.max(magnitudes_db)),
        cmap=image.cmap
    )

    cbar = axs.figure.colorbar(mappable=mappable, ax=axs)
    cbar.ax.set_ylabel("(dbV)")

    if cutoff is not None:
        color = (0.0, 0.75, 0.0)

        axs.axhline(y=cutoff, color=color, linestyle="dashed", linewidth=3)
        axs.axhline(y=-float(cutoff), color=color, linestyle="dashed", linewidth=3)

    axs.set_xlabel(f"Window ({spectrogram.shape[0]} timestep/window)")
    axs.set_ylabel(r"$\omega$ (cycles/sample)")

    #***********************************************************************************************
    # Set the y-ticks.
    #***********************************************************************************************
    axs.set_yticks(np.arange(start=-0.5, stop=0.55, step=0.1))

    return (fig, axs)
