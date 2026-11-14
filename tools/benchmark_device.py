"""Comprehensive device performance benchmark for Edge Poultry Counter Service.

Measures hardware specs, video decoding throughput, detector/tracker latency,
CPU/memory utilization, and concurrent stream capacity on edge devices.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any
import cv2
import numpy as np
import psutil

# Add service root to sys.path
SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from src.capture.source import FileVideoSource, SyntheticVideoSource
from src.detection.detector import MockDetector, ShackleDetector
from src.tracking.tracker import ChickenCounter, VirtualLineZone


def get_system_specs() -> dict[str, Any]:
    """Collect host system and hardware specifications."""
    mem = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "processor": platform.processor() or "Unknown CPU",
        "cpu_count_physical": psutil.cpu_count(logical=False) or os.cpu_count() or 1,
        "cpu_count_logical": os.cpu_count() or 1,
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_available_gb": round(mem.available / (1024**3), 2),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }


def benchmark_pipeline(
    source: Any,
    detector: Any,
    line_zone: VirtualLineZone,
    num_frames: int = 150,
) -> dict[str, Any]:
    """Execute unthrottled pipeline benchmark measuring latency, FPS, and resource utilization."""
    counter = ChickenCounter(line_zone=line_zone, line_id="bench-line")
    process = psutil.Process(os.getpid())

    # Warmup
    source.start()
    for _ in range(min(10, num_frames)):
        ret, frame = source.read()
        if not ret or frame is None:
            break
        detections = detector.predict(frame)
        counter.update(detections, timestamp=time.time())

    # Measurement
    latencies: list[float] = []
    mem_before = process.memory_info().rss / (1024 * 1024)

    t_start = time.perf_counter()
    cpu_start = process.cpu_times()

    frames_processed = 0
    for _ in range(num_frames):
        f_start = time.perf_counter()
        ret, frame = source.read()
        if not ret or frame is None:
            break

        detections = detector.predict(frame)
        counter.update(detections, timestamp=time.time())

        f_lat = (time.perf_counter() - f_start) * 1000.0
        latencies.append(f_lat)
        frames_processed += 1

    total_wall_time = max(0.0001, time.perf_counter() - t_start)
    cpu_end = process.cpu_times()
    mem_after = process.memory_info().rss / (1024 * 1024)
    source.stop()

    if not latencies:
        return {"error": "No frames processed"}

    # Compute CPU usage across the measurement period
    cpu_time_used = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    cpu_pct = (cpu_time_used / total_wall_time) * 100.0

    avg_fps = frames_processed / total_wall_time
    sorted_lat = sorted(latencies)

    return {
        "frames_processed": frames_processed,
        "total_duration_sec": round(total_wall_time, 3),
        "fps": round(avg_fps, 1),
        "latency_min_ms": round(min(latencies), 2),
        "latency_avg_ms": round(sum(latencies) / len(latencies), 2),
        "latency_median_ms": round(float(np.median(latencies)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "latency_max_ms": round(max(latencies), 2),
        "chickens_counted": counter.total_count,
        "process_cpu_pct": round(cpu_pct, 1),
        "ram_delta_mb": round(mem_after - mem_before, 2),
        "ram_rss_mb": round(mem_after, 2),
    }


def run_device_benchmark(
    num_frames: int = 150,
    video_path: str | None = None,
    output_json: str | None = None,
) -> dict[str, Any]:
    """Run full device benchmark across synthetic and real video workloads."""
    print("=" * 74)
    print("            EDGE DEVICE PERFORMANCE BENCHMARK SUITE")
    print("=" * 74)

    specs = get_system_specs()
    print(f"OS:               {specs['os']}")
    print(f"Processor:        {specs['processor']}")
    print(f"Cores:            {specs['cpu_count_physical']} Physical / {specs['cpu_count_logical']} Logical")
    print(f"System RAM:       {specs['ram_total_gb']} GB Total ({specs['ram_available_gb']} GB Free)")
    print(f"Runtime:          Python {specs['python_version']} | OpenCV {specs['opencv_version']} | NumPy {specs['numpy_version']}")
    print("=" * 74)

    results: dict[str, Any] = {"specs": specs, "benchmarks": {}}

    # Benchmark 1: Synthetic 640x480 with MockDetector
    print(f"\n[Test 1/3] Benchmarking Mock Detector (Synthetic 640x480, {num_frames} frames)...")
    source_synth_mock = SyntheticVideoSource(width=640, height=480, fps=30)
    line_synth = VirtualLineZone(start=(0, 240), end=(640, 240), direction="down")
    res_synth_mock = benchmark_pipeline(
        source_synth_mock,
        MockDetector(min_confidence=0.8),
        line_synth,
        num_frames=num_frames,
    )
    results["benchmarks"]["synthetic_mock_640x480"] = res_synth_mock
    print(f"  -> FPS: {res_synth_mock['fps']} fps | Latency p95: {res_synth_mock['latency_p95_ms']} ms | CPU: {res_synth_mock['process_cpu_pct']}%")

    # Benchmark 2: Synthetic 640x480 with ShackleDetector
    print(f"\n[Test 2/3] Benchmarking Shackle Detector (Synthetic 640x480, {num_frames} frames)...")
    source_synth_shackle = SyntheticVideoSource(width=640, height=480, fps=30)
    res_synth_shackle = benchmark_pipeline(
        source_synth_shackle,
        ShackleDetector(min_confidence=0.5),
        line_synth,
        num_frames=num_frames,
    )
    results["benchmarks"]["synthetic_shackle_640x480"] = res_synth_shackle
    print(f"  -> FPS: {res_synth_shackle['fps']} fps | Latency p95: {res_synth_shackle['latency_p95_ms']} ms | CPU: {res_synth_shackle['process_cpu_pct']}%")

    # Benchmark 3: Real Video File (if provided or exists in samples/)
    target_video = video_path
    if not target_video:
        default_sample = SERVICE_ROOT / "samples" / "Rekaman_2.MOV"
        if default_sample.exists():
            target_video = str(default_sample)

    if target_video and Path(target_video).exists():
        cap = cv2.VideoCapture(target_video)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"\n[Test 3/3] Benchmarking Industrial Video Pipeline ({Path(target_video).name} [{vw}x{vh}], {num_frames} frames)...")
        source_file = FileVideoSource(target_video, loop=True)
        line_video = VirtualLineZone(start=(vw // 2, int(vh * 0.2)), end=(vw // 2, int(vh * 0.9)), direction="right")
        res_real_video = benchmark_pipeline(
            source_file,
            ShackleDetector(min_confidence=0.5),
            line_video,
            num_frames=num_frames,
        )
        res_real_video["resolution"] = f"{vw}x{vh}"
        res_real_video["file"] = Path(target_video).name
        results["benchmarks"]["real_conveyor_video"] = res_real_video
        print(f"  -> FPS: {res_real_video['fps']} fps | Latency p95: {res_real_video['latency_p95_ms']} ms | CPU: {res_real_video['process_cpu_pct']}%")
    else:
        print("\n[Test 3/3] Real video sample not found. Skipping file decode benchmark.")

    # Capacity Analysis
    print("\n" + "=" * 74)
    print("                    DEVICE CAPACITY & SIZING SUMMARY")
    print("=" * 74)

    target_feed_fps = 30.0
    sla_latency_target_ms = 50.0

    primary_bench = results["benchmarks"].get("real_conveyor_video") or res_synth_shackle
    fps = primary_bench["fps"]
    p95 = primary_bench["latency_p95_ms"]
    cpu = primary_bench["process_cpu_pct"]

    meets_fps = fps >= target_feed_fps
    meets_lat = p95 <= sla_latency_target_ms

    print(f"  Primary Workload Tested:    {primary_bench.get('file', 'Synthetic 640x480')} ({primary_bench.get('resolution', '640x480')})")
    print(f"  Throughput Achieved:        {fps:.1f} FPS (Target: >= {target_feed_fps:.1f} FPS) -> {'PASS [OK]' if meets_fps else 'FAIL [BELOW TARGET]'}")
    print(f"  95th-Percentile Latency:    {p95:.2f} ms (Target: <= {sla_latency_target_ms:.1f} ms) -> {'PASS [OK]' if meets_lat else 'FAIL [SLOW]'}")
    print(f"  Single Stream CPU Overhead: {cpu:.1f}% of a single core")

    # Estimated multi-stream capacity assuming 70% total system CPU ceiling
    total_cpu_budget_pct = 70.0 * specs["cpu_count_logical"]
    # CPU overhead per line in full system %
    cpu_per_line_sys_pct = max(1.0, cpu)
    est_lines_by_cpu = int(total_cpu_budget_pct // cpu_per_line_sys_pct)
    # Also bounded by single core throughput ratio
    est_lines_by_fps = max(1, int(fps // target_feed_fps))

    print("\n  Estimated Multi-Line Capacity on this Device:")
    print(f"    - Maximum Concurrent Real-Time 30 FPS Feeds: ~{max(1, est_lines_by_fps)} line(s) per core")
    print(f"    - Theoretical Multi-Process Capacity:        ~{max(1, est_lines_by_cpu)} line(s) (at 70% CPU cap across {specs['cpu_count_logical']} cores)")

    overall_tier = "High-End Workstation / Server" if fps > 100 else ("Production Edge Ready" if fps >= 30 else "Insufficient for 30 FPS")
    print(f"\n  Device Performance Tier:    {overall_tier}")
    print("=" * 74 + "\n")

    results["assessment"] = {
        "meets_fps_sla": meets_fps,
        "meets_latency_sla": meets_lat,
        "est_concurrent_lines": est_lines_by_cpu,
        "device_tier": overall_tier,
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results exported to: {output_json}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge Device Hardware Performance Benchmark")
    parser.add_argument("--frames", type=int, default=150, help="Frames per benchmark test (default: 150)")
    parser.add_argument("--video", type=str, default=None, help="Path to custom conveyor video file")
    parser.add_argument("--json", type=str, default=None, help="Output path for JSON results export")
    args = parser.parse_args()

    run_device_benchmark(
        num_frames=args.frames,
        video_path=args.video,
        output_json=args.json,
    )


if __name__ == "__main__":
    main()
