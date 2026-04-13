"""integration_test.py — Verify io and viz modules work together.

Run from the project root: python integration_test.py
"""
import os
from seismoai_io import load_sgy, load_folder, normalize_traces
from seismoai_viz import plot_gather, plot_trace, plot_spectrum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Step 1: Load a single file ----
print("Step 1: Loading single SGY file...")
data = load_sgy("data/27_1511546140_30100_50100_20171127_150416_752.sgy")
print(f"  Traces: {data['n_traces']}, Samples: {data['n_samples']}")
print(f"  Sample rate: {data['sample_rate_ms']} ms")
print(f"  Amplitude range: {data['traces'].min():.2f} to {data['traces'].max():.2f}")

# ---- Step 2: Load folder ----
print("\nStep 2: Loading all files from data/ folder...")
all_data = load_folder("data")
print(f"  Loaded {len(all_data)} files")

# ---- Step 3: Normalize ----
print("\nStep 3: Normalizing traces...")
normed = normalize_traces(data['traces'], method='zscore')
print(f"  Normalized mean: {normed.mean():.8f} (should be ~0)")
print(f"  Normalized std:  {normed.std():.4f}")

# ---- Step 4: Create output folder ----
os.makedirs("output", exist_ok=True)

# ---- Step 5: Plot gather (raw) ----
print("\nStep 4: Plotting raw seismic gather...")
fig1 = plot_gather(data['traces'],
                   title="Raw Seismic Gather - Shot 27",
                   save_path="output/gather_raw.png")
plt.close(fig1)
print("  Saved: output/gather_raw.png")

# ---- Step 6: Plot gather (normalized) ----
print("Step 5: Plotting normalized gather...")
fig2 = plot_gather(normed,
                   title="Normalized Gather (z-score) - Shot 27",
                   save_path="output/gather_normalized.png")
plt.close(fig2)
print("  Saved: output/gather_normalized.png")

# ---- Step 7: Plot single trace ----
print("Step 6: Plotting trace 50 waveform...")
fig3 = plot_trace(data['traces'][50],
                  sample_rate_ms=data['sample_rate_ms'],
                  trace_index=50,
                  save_path="output/trace_50.png")
plt.close(fig3)
print("  Saved: output/trace_50.png")

# ---- Step 8: Plot spectrum ----
print("Step 7: Plotting spectrum of trace 50...")
fig4 = plot_spectrum(data['traces'][50],
                     sample_rate_ms=data['sample_rate_ms'],
                     title="Spectrum - Trace 50",
                     save_path="output/spectrum_50.png")
plt.close(fig4)
print("  Saved: output/spectrum_50.png")

print("\n" + "="*50)
print("ALL INTEGRATION TESTS PASSED!")
print("Check the output/ folder for generated plots.")
print("="*50)
