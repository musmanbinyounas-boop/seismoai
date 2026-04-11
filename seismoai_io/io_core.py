"""SeismoAI I/O module — load and prepare SGY seismic files.

This module provides functions to:
1. Load a single .sgy file (load_sgy)
2. Load all .sgy files from a folder (load_folder)
3. Normalize trace amplitudes (normalize_traces)

The functions handle the Utah FORGE dataset which uses little-endian
byte order (format code 5 = IEEE float).
"""

import os
import glob
import numpy as np
import pandas as pd
import segyio


def load_sgy(filepath: str) -> dict:
    """Load a single SGY file and return traces + metadata.

    This function opens a SEG-Y file, reads all seismic traces
    (the actual amplitude data) and trace headers (metadata like
    position and offset), and returns them in a convenient format.

    Parameters
    ----------
    filepath : str
        Path to the .sgy file on disk.

    Returns
    -------
    dict with keys:
        'traces'         : np.ndarray of shape (n_traces, n_samples)
                           The raw seismic amplitude data.
        'headers'        : pd.DataFrame with one row per trace.
                           Contains all 91 standard SEG-Y header fields.
        'sample_rate_ms' : float, the time between samples (1.0 ms
                           for our dataset).
        'n_traces'       : int, number of traces in the file.
        'n_samples'      : int, number of samples per trace.
        'filepath'       : str, the original file path (for reference).

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    RuntimeError
        If the file cannot be parsed as valid SEG-Y with either endianness.

    Examples
    --------
    >>> data = load_sgy("data/27_1511546140_30100_50100_20171127_150416_752.sgy")
    >>> print(data['traces'].shape)
    (167, 4001)
    >>> print(data['sample_rate_ms'])
    1.0
    """
    # Step 1: Check the file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"SGY file not found: {filepath}")

    # Step 2: Try to open with segyio
    # WHY two endians? SEG-Y files can be big-endian (older standard)
    # or little-endian. Our FORGE data is little-endian, but we try
    # both so this function works with ANY sgy file.
    for endian in ('little', 'big'):
        try:
            with segyio.open(filepath, ignore_geometry=True,
                             endian=endian) as f:
                # Read ALL traces into a 2D numpy array
                # Shape: (number_of_traces, samples_per_trace)
                traces = segyio.tools.collect(f.trace[:])

                # Get sample interval in milliseconds
                # segyio returns microseconds, so divide by 1000
                sample_rate_ms = segyio.tools.dt(f) / 1000.0

                # Read all trace headers into a pandas DataFrame
                # Each trace has 91 standard header fields (position,
                # offset, gain, etc.)
                headers = pd.DataFrame([
                    {str(k): v for k, v in f.header[i].items()}
                    for i in range(f.tracecount)
                ])

            # If we get here without error, the endian was correct
            return {
                'traces': traces,
                'headers': headers,
                'sample_rate_ms': sample_rate_ms,
                'n_traces': traces.shape[0],
                'n_samples': traces.shape[1],
                'filepath': filepath,
            }
        except RuntimeError:
            # Wrong endianness — try the other one
            continue

    # If both endians failed, the file is probably corrupt
    raise RuntimeError(
        f"Cannot open SGY file (tried both endians): {filepath}"
    )


def load_folder(folder_path: str) -> list:
    """Load all .sgy files from a folder.

    This is a convenience function that calls load_sgy() on every
    .sgy file found in the given directory. Files that fail to load
    are skipped with a warning (instead of crashing the whole batch).

    Parameters
    ----------
    folder_path : str
        Path to a directory containing .sgy files.

    Returns
    -------
    list of dict
        Each element has the same structure as load_sgy() output.
        The list is sorted alphabetically by filename.

    Raises
    ------
    FileNotFoundError
        If the folder doesn't exist or contains no .sgy files.

    Examples
    --------
    >>> datasets = load_folder("data/")
    >>> print(len(datasets))
    2
    >>> print(datasets[0]['n_traces'])
    167
    """
    # Step 1: Check the folder exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Step 2: Find all .sgy files, sorted alphabetically
    files = sorted(glob.glob(os.path.join(folder_path, "*.sgy")))
    if not files:
        raise FileNotFoundError(
            f"No .sgy files found in: {folder_path}"
        )

    # Step 3: Load each file, skip failures
    results = []
    for fp in files:
        try:
            results.append(load_sgy(fp))
            print(f"  Loaded: {os.path.basename(fp)}")
        except RuntimeError as e:
            print(f"  WARNING: Skipping {os.path.basename(fp)}: {e}")

    return results

