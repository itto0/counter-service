"""Tests for benchmark harness."""

from unittest.mock import patch
from benchmark import main, run_benchmark
from tools.benchmark_device import get_system_specs, run_device_benchmark, main as device_benchmark_main


def test_run_benchmark_metrics():
    """Verify run_benchmark processes frames and meets performance criteria."""
    metrics = run_benchmark(num_frames=60)

    assert metrics["frames_processed"] == 60
    assert metrics["avg_fps"] >= 30.0
    assert metrics["p95_lat_ms"] < 50.0
    assert metrics["total_count"] >= 0
    assert metrics["total_time_sec"] > 0


def test_benchmark_cli_main():
    """Verify benchmark CLI parses args and runs benchmark."""
    with patch("sys.argv", ["benchmark.py", "--frames", "20"]):
        # Should not raise exception
        main()


def test_device_benchmark_system_specs():
    """Verify device benchmark extracts system specs properly."""
    specs = get_system_specs()
    assert "os" in specs
    assert "processor" in specs
    assert specs["cpu_count_logical"] >= 1
    assert specs["ram_total_gb"] > 0


def test_device_benchmark_run(tmp_path):
    """Verify run_device_benchmark runs and exports json."""
    out_json = str(tmp_path / "bench_out.json")
    results = run_device_benchmark(num_frames=10, output_json=out_json)

    assert "specs" in results
    assert "benchmarks" in results
    assert "synthetic_mock_640x480" in results["benchmarks"]
    assert "assessment" in results
    assert (tmp_path / "bench_out.json").exists()


def test_device_benchmark_cli(tmp_path):
    """Verify device benchmark CLI runs without errors."""
    out_json = str(tmp_path / "cli_bench.json")
    with patch("sys.argv", ["tools/benchmark_device.py", "--frames", "10", "--json", out_json]):
        device_benchmark_main()
    assert (tmp_path / "cli_bench.json").exists()

