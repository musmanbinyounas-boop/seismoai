# SeismoAI — Seismic Data Processing Library

A Python library for loading, visualizing, and analyzing seismic data
from SEG-Y files. Built as part of the MLOps course project.

## Dataset

Utah FORGE 2D Seismic Survey (2017):
- 166 SGY files, each with 167 traces x 4,001 samples
- 1 ms sample interval (4 seconds per trace)
- Little-endian byte order (IEEE float format)

## Installation

\`\`\`bash
pip install seismoai-io-USM-JZB
pip install seismoai_viz #created by Jawad Ali
\`\`\`

## Quick Start

\`\`\`python
from seismoai_io import load_sgy, normalize_traces
from seismoai_viz import plot_gather, plot_trace, plot_spectrum

# Load a seismic file
data = load_sgy("path/to/file.sgy")
print(f"Loaded {data['n_traces']} traces")

# Normalize amplitudes
normed = normalize_traces(data['traces'], method='zscore')

# Visualize
fig = plot_gather(normed, title="My Seismic Data")
fig = plot_trace(data['traces'][50], trace_index=50)
fig = plot_spectrum(data['traces'][50])
\`\`\`

## Modules

### seismoai_io
- \`load_sgy(filepath)\` — Load a single .sgy file
- \`load_folder(folder_path)\` — Load all .sgy files from a directory
- \`normalize_traces(traces, method)\` — Normalize amplitudes

### seismoai_viz
- \`plot_gather(traces, ...)\` — 2D seismic gather image
- \`plot_trace(trace, ...)\` — Single trace waveform
- \`plot_spectrum(trace, ...)\` — Frequency spectrum (FFT)

## Running Tests

\`\`\`bash
pytest seismoai_io/tests/ -v
pytest seismoai_viz/tests/ -v
\`\`\`

## Team

- Member 1 (M Usman Younas) — seismoai_io: load_sgy, load_folder
- Member 2 (Jazib Noel) — seismoai_io: normalize_traces
- Member 3 (Jawad Ali) — seismoai_viz: plot_gather, plot_trace
- Member 4 (Wasif Ali Pervez) — seismoai_viz: plot_spectrum

## License

MIT
