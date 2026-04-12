"""SeismoAI visualization module — plot seismic gathers and spectra.

This module provides three plotting functions:
1. plot_gather  — 2D image of all traces (the standard seismic view)
2. plot_trace   — waveform of a single trace
3. plot_spectrum — frequency content of a single trace (FFT)

All functions return a matplotlib Figure object so you can further
customize the plot if needed. All support an optional save_path
parameter to save the plot as an image file.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
# WHY 'Agg'? This backend doesn't need a display/screen,
# so it works in terminals, servers, CI/CD, and automated tests.
# Without it, matplotlib may crash on headless systems.
import matplotlib.pyplot as plt


def plot_gather(traces: np.ndarray,
                sample_rate_ms: float = 1.0,
                title: str = "Seismic Gather",
                cmap: str = "seismic",
                clip_percentile: float = 99.0,
                save_path: str = None) -> plt.Figure:
    """Plot seismic traces as a 2D gather image.

    A seismic gather is the standard way geophysicists view data:
    - X-axis = trace number (each trace is one sensor recording)
    - Y-axis = time (going downward, as is convention in geophysics)
    - Color = amplitude (red = positive, blue = negative with
      'seismic' colormap)

    The clip_percentile parameter is crucial: our data has outlier
    amplitudes up to 758, while most values are between -1 and +1.
    Without clipping, the image would look blank because the color
    scale would be dominated by the outliers.

    Parameters
    ----------
    traces : np.ndarray
        2D array of shape (n_traces, n_samples).
    sample_rate_ms : float
        Time between consecutive samples in milliseconds.
    title : str
        Title displayed at the top of the plot.
    cmap : str
        Matplotlib colormap name. 'seismic' is the geophysics
        standard (blue-white-red). Other options: 'gray', 'RdBu'.
    clip_percentile : float
        Clip amplitudes at this percentile. 99.0 means values above
        the 99th percentile are capped. Lower = more clipping.
    save_path : str or None
        If provided, saves the plot to this file path (e.g. 'plot.png').

    Returns
    -------
    matplotlib.figure.Figure
        The figure object. Call plt.close(fig) when done to free memory.

    Examples
    --------
    >>> from seismoai_io import load_sgy
    >>> data = load_sgy("data/27_...sgy")
    >>> fig = plot_gather(data['traces'], title="Shot 27")
    """
    n_traces, n_samples = traces.shape

    # Convert sample indices to time in seconds
    # e.g., 4001 samples at 1ms = 0.000, 0.001, ..., 4.000 seconds
    time_axis = np.arange(n_samples) * sample_rate_ms / 1000.0

    # CLIPPING: Calculate the amplitude threshold
    # np.percentile(|traces|, 99) finds the value below which 99%
    # of absolute amplitudes fall. We cap all values at this level.
    clip_val = np.percentile(np.abs(traces), clip_percentile)
    if clip_val == 0:
        clip_val = 1.0  # Avoid clipping to zero for dead gathers
    clipped = np.clip(traces, -clip_val, clip_val)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # extent = [left, right, bottom, top] of the image
    # We put time going downward (top=0, bottom=max_time)
    extent = [0, n_traces - 1, time_axis[-1], time_axis[0]]

    ax.imshow(clipped.T,  # Transpose: rows=samples, columns=traces
              aspect='auto',
              cmap=cmap,
              extent=extent,
              interpolation='bilinear')

    ax.set_xlabel("Trace Number")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)

    # Add a colorbar to show the amplitude-color mapping
    plt.colorbar(ax.images[0], ax=ax, label="Amplitude")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_trace(trace: np.ndarray,
               sample_rate_ms: float = 1.0,
               trace_index: int = 0,
               title: str = None,
               save_path: str = None) -> plt.Figure:
    """Plot a single seismic trace as a waveform.

    Shows amplitude vs time, like an oscilloscope or ECG display.
    Useful for examining individual traces to see signal quality,
    check for noise, or identify arrivals (the moment a wave
    reaches the sensor).

    Parameters
    ----------
    trace : np.ndarray
        1D array of amplitude samples for one trace.
    sample_rate_ms : float
        Time between consecutive samples in milliseconds.
    trace_index : int
        The trace number (used in default title for identification).
    title : str or None
        Custom title. If None, defaults to "Trace {trace_index}".
    save_path : str or None
        If provided, saves the plot to this file path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from seismoai_io import load_sgy
    >>> data = load_sgy("data/27_...sgy")
    >>> fig = plot_trace(data['traces'][50], trace_index=50)
    """
    n_samples = trace.shape[0]
    time_axis = np.arange(n_samples) * sample_rate_ms / 1000.0

    if title is None:
        title = f"Trace {trace_index}"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, trace,
            linewidth=0.5,
            color='black')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


