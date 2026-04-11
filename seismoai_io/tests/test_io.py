"""Tests for seismoai_io module.

Run these with: pytest seismoai_io/tests/test_io.py -v
The -v flag shows verbose output (each test name + pass/fail).
"""
import os
import pytest
import numpy as np
from seismoai_io import load_sgy, load_folder, normalize_traces

# Path to a sample SGY file for testing
# Update this if your file is in a different location
SAMPLE_SGY = os.path.join(
    "data",
    "27_1511546140_30100_50100_20171127_150416_752.sgy"
)


# ============================================================
# Tests for load_sgy()
# ============================================================
class TestLoadSgy:
    """Tests for the load_sgy function."""

    def test_loads_valid_file(self):
        """Test that load_sgy correctly loads a real SGY file."""
        if not os.path.isfile(SAMPLE_SGY):
            pytest.skip("Sample SGY file not available")
        result = load_sgy(SAMPLE_SGY)
        # Our files have 167 traces with 4001 samples each
        assert result['n_traces'] == 167
        assert result['n_samples'] == 4001
        assert result['sample_rate_ms'] == 1.0
        assert result['traces'].shape == (167, 4001)

    def test_returns_headers_dataframe(self):
        """Test that headers is a DataFrame with one row per trace."""
        if not os.path.isfile(SAMPLE_SGY):
            pytest.skip("Sample SGY file not available")
        result = load_sgy(SAMPLE_SGY)
        assert len(result['headers']) == 167

    def test_file_not_found(self):
        """Test that a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_sgy("this_file_does_not_exist.sgy")


# ============================================================
# Tests for load_folder()
# ============================================================
class TestLoadFolder:
    """Tests for the load_folder function."""

    def test_loads_folder(self):
        """Test that load_folder returns a list of datasets."""
        if not os.path.isdir("data"):
            pytest.skip("Data folder not available")
        results = load_folder("data")
        assert len(results) > 0
        assert results[0]['n_traces'] == 167

    def test_folder_not_found(self):
        """Test that a missing folder raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_folder("this_folder_does_not_exist")


# ============================================================
# Tests for normalize_traces()
# ============================================================
class TestNormalizeTraces:
    """Tests for the normalize_traces function."""

    def test_zscore_produces_zero_mean_unit_std(self):
        """After zscore normalization, each trace should have
        mean approximately 0 and std approximately 1."""
        traces = np.random.randn(5, 100).astype(np.float32)
        normed = normalize_traces(traces, method='zscore')
        for i in range(5):
            assert abs(normed[i].mean()) < 1e-10
            assert abs(normed[i].std() - 1.0) < 1e-10

    def test_minmax_produces_01_range(self):
        """After minmax normalization, values should be in [0, 1]."""
        traces = np.random.randn(5, 100).astype(np.float32)
        normed = normalize_traces(traces, method='minmax')
        for i in range(5):
            assert normed[i].min() >= -1e-10
            assert normed[i].max() <= 1.0 + 1e-10

    def test_trace_max_produces_minus1_to_1(self):
        """After trace_max normalization, the peak value should be
        +1 or -1."""
        traces = np.array([[0, 5, -10, 3]], dtype=np.float32)
        normed = normalize_traces(traces, method='trace_max')
        assert abs(normed[0, 2] - (-1.0)) < 1e-10

    def test_dead_trace_stays_zero(self):
        """A trace of all zeros should remain all zeros (no crash)."""
        traces = np.zeros((1, 100), dtype=np.float32)
        normed = normalize_traces(traces, method='zscore')
        assert np.all(normed == 0.0)

    def test_invalid_method_raises_error(self):
        """An unrecognized method name should raise ValueError."""
        with pytest.raises(ValueError):
            normalize_traces(np.zeros((2, 10)), method='bad')
