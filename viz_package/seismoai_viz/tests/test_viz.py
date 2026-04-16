"""Tests for seismoai_viz module.

Run these with: pytest seismoai_viz/tests/test_viz.py -v
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from seismoai_viz import plot_gather, plot_trace, plot_spectrum


class TestPlotGather:
    """Tests for the plot_gather function."""

    def test_returns_figure(self):
        """plot_gather should return a matplotlib Figure."""
        traces = np.random.randn(20, 500).astype(np.float32)
        fig = plot_gather(traces, sample_rate_ms=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        """plot_gather should save a PNG when save_path is given.
        tmp_path is a pytest fixture that creates a temporary directory."""
        traces = np.random.randn(20, 500).astype(np.float32)
        path = str(tmp_path / "gather.png")
        fig = plot_gather(traces, save_path=path)
        assert os.path.isfile(path)
        plt.close(fig)

    def test_handles_outliers(self):
        """plot_gather should not crash when traces have extreme values."""
        traces = np.random.randn(10, 100).astype(np.float32)
        traces[0, 0] = 1000.0  # simulated outlier
        fig = plot_gather(traces, clip_percentile=95.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTrace:
    """Tests for the plot_trace function."""

    def test_returns_figure(self):
        """plot_trace should return a matplotlib Figure."""
        trace = np.sin(np.linspace(0, 10, 500))
        fig = plot_trace(trace, sample_rate_ms=2.0, trace_index=5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self):
        """Custom title should appear on the plot."""
        trace = np.zeros(100)
        fig = plot_trace(trace, title="My Custom Title")
        assert fig.axes[0].get_title() == "My Custom Title"
        plt.close(fig)

    def test_default_title(self):
        """Default title should include trace index."""
        trace = np.zeros(100)
        fig = plot_trace(trace, trace_index=42)
        assert "42" in fig.axes[0].get_title()
        plt.close(fig)


class TestPlotSpectrum:
    """Tests for the plot_spectrum function."""

    def test_returns_figure_with_two_subplots(self):
        """plot_spectrum should return a Figure with 2 axes
        (linear and dB)."""
        t = np.arange(4001) * 0.001  # 4001 samples at 1ms
        trace = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine
        fig = plot_spectrum(trace, sample_rate_ms=1.0)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        """plot_spectrum should save a PNG when save_path is given."""
        trace = np.random.randn(1000)
        path = str(tmp_path / "spectrum.png")
        fig = plot_spectrum(trace, save_path=path)
        assert os.path.isfile(path)
        plt.close(fig)
