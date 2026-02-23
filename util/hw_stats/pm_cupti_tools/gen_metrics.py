#!/usr/bin/env python3
"""
CUPTI PM Sampling Metric Tool

Default behavior:
    Queries available metrics using ncu --query-metrics-mode all --query-metrics-collection pmsampling
    and saves to <output-dir>/<gpu_name>/pmsampling_metrics.txt

With --sweep flag:
    1. Queries available metrics from ncu --query-metrics
    2. Gets the GPU name and creates output folder based on it
    3. Sweeps through Counter metrics, testing each with PM sampling injection
    4. Records which metrics succeed/fail

Usage:
    python3 gen_metrics.py                    # Query and save pmsampling metrics (default)
    python3 gen_metrics.py --sweep            # Run the full metric sweep
    python3 gen_metrics.py --output-dir DIR   # Specify output directory
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def get_gpu_info() -> Tuple[str, str]:
    """Get GPU name and chip name from ncu."""
    try:
        result = subprocess.run(
            ["ncu", "--query-metrics"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "TMPDIR": "/tmp"}
        )

        # Parse device line: "Device Tesla V100-SXM2-32GB (GV100)"
        for line in result.stdout.split('\n') + result.stderr.split('\n'):
            match = re.match(r'Device\s+(.+?)\s+\((\w+)\)', line)
            if match:
                gpu_name = match.group(1).strip()
                chip_name = match.group(2).strip()
                return gpu_name, chip_name

        raise RuntimeError("Could not parse GPU name from ncu output")

    except subprocess.TimeoutExpired:
        raise RuntimeError("ncu --query-metrics timed out")
    except FileNotFoundError:
        raise RuntimeError("ncu not found. Make sure Nsight Compute is in PATH")


def parse_metrics_output(output: str) -> Tuple[Dict, List[Dict]]:
    """
    Parse ncu --query-metrics output into structured data, grouping variants.

    Returns (device_info, metrics_list) where:
    - device_info: {"name": str, "chip": str}
    - metrics_list: list of {"name", "type", "unit", "domains", "description", "variants"}

    Metrics like crop__cycles_active.avg.peak_sustained_active are grouped under
    their base metric crop__cycles_active.avg with "peak_sustained_active" as a variant.
    """
    device_info = {}
    raw_metrics = []

    for line in output.split('\n'):
        # Parse device line: "Device NVIDIA GeForce RTX 5090 (GB202)"
        device_match = re.match(r'Device\s+(.+?)\s+\((\w+)\)', line)
        if device_match:
            device_info = {
                "name": device_match.group(1).strip(),
                "chip": device_match.group(2).strip()
            }
            continue

        # Skip header and separator lines
        if not line.strip() or line.startswith('-') or line.startswith('Metric Name'):
            continue

        # Parse metric lines using pattern matching (columns can have variable width)
        # Format: MetricName   Type   Unit   Domains   Description
        # Description typically starts with "# of" or similar
        if re.match(r'^[a-z][a-z0-9_]*__', line):
            # Use regex to parse the line
            # Pattern: name, then type (Counter/Ratio/Throughput), then unit, then domains, then description
            match = re.match(
                r'^(\S+)\s+'                           # metric name
                r'(Counter|Ratio|Throughput)\s+'       # type
                r'(\S*)\s+'                            # unit (can be empty, match non-space)
                r'(\[?[\w,_]+\]?)\s+'                  # domains (can be [a,b,c] or single)
                r'(.*)$',                              # description
                line
            )
            if match:
                metric = {
                    "name": match.group(1),
                    "type": match.group(2),
                    "unit": match.group(3) if match.group(3) else "",
                    "domains": match.group(4),
                    "description": match.group(5).strip(),
                }
                raw_metrics.append(metric)

    # Group metrics by base name
    # Base metrics end with .avg, .sum, .min, .max, .ratio, etc.
    # Variants add suffixes like .peak_sustained_active, .per_second, etc.
    base_metric_suffixes = {'avg', 'sum', 'min', 'max', 'ratio', 'pct', 'per_cycle_elapsed'}

    # First pass: identify all base metric names
    base_names = set()
    for m in raw_metrics:
        name = m["name"]
        parts = name.split('.')
        # Find the base: unit__metric.aggregation
        for i in range(2, len(parts) + 1):
            candidate = '.'.join(parts[:i])
            if parts[i-1] in base_metric_suffixes:
                base_names.add(candidate)
                break
        else:
            # If no known suffix found, treat the shortest form as base
            # (happens when metric has only 2 parts like unit__name.type)
            if len(parts) >= 2:
                base_names.add('.'.join(parts[:2]))

    # Second pass: for each metric, find its base and variant
    metrics_grouped = {}
    for m in raw_metrics:
        name = m["name"]

        # Find the longest matching base name
        base_name = None
        for candidate in sorted(base_names, key=len, reverse=True):
            if name == candidate or name.startswith(candidate + '.'):
                base_name = candidate
                break

        if not base_name:
            base_name = name

        # Extract variant (everything after base name)
        if name == base_name:
            variant = None
        else:
            variant = name[len(base_name) + 1:]  # +1 for the dot

        if base_name not in metrics_grouped:
            # Extract base description (remove variant-specific text in parentheses)
            desc = m.get("description", "")
            # Clean description: extract the main part before "(This counter metric..."
            desc_match = re.match(r'^(.+?)\s*\(This\s+\w+\s+metric', desc)
            base_desc = desc_match.group(1).strip() if desc_match else desc

            # Parse domains - can be comma-separated, possibly in brackets [a,b,c]
            domains_str = m["domains"].strip('[]')
            domains = [d.strip() for d in domains_str.split(',') if d.strip()]

            metrics_grouped[base_name] = {
                "name": base_name,
                "type": m["type"],
                "unit": m["unit"],
                "domains": domains,
                "description": base_desc,
                "variants": []
            }

        if variant:
            metrics_grouped[base_name]["variants"].append(variant)

    # Convert to sorted list
    metrics = [metrics_grouped[k] for k in sorted(metrics_grouped.keys())]

    return device_info, metrics


def query_pmsampling_metrics(output_dir: str = "metrics") -> None:
    """
    Query metrics with pmsampling collection mode and save to GPU-specific folder.

    Runs: ncu --query-metrics-mode all --query-metrics-collection pmsampling
    """
    print("Detecting GPU...")
    try:
        gpu_name, chip_name = get_gpu_info()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"GPU: {gpu_name} ({chip_name})")

    # Create output directory based on GPU name (same as sweep mode)
    gpu_folder_name = gpu_name.replace(' ', '_').replace('-', '_')
    output_path = Path(output_dir) / gpu_folder_name
    output_path.mkdir(parents=True, exist_ok=True)

    print("Querying PM sampling metrics from ncu...")

    try:
        result = subprocess.run(
            ["ncu", "--query-metrics-mode", "all", "--query-metrics-collection", "pmsampling"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "TMPDIR": "/tmp"}
        )

        # Combine stdout and stderr as ncu may output to either
        all_output = result.stdout + result.stderr

        # Parse the output
        device_info, metrics = parse_metrics_output(all_output)

        # Save YAML file with blank lines between metrics for readability
        yaml_file = output_path / "pmsampling_metrics.yaml"
        with open(yaml_file, 'w') as f:
            # Write header section
            header = {
                "device": device_info,
                "metric_count": len(metrics),
            }
            yaml.dump(header, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

            # Write metrics with blank lines between each
            f.write("metrics:\n")
            for metric in metrics:
                f.write("\n")  # Empty line before each metric
                metric_yaml = yaml.dump([metric], default_flow_style=False, sort_keys=False, allow_unicode=True)
                f.write(metric_yaml)

        print(f"Saved PM sampling metrics to: {yaml_file}")
        print(f"Found {len(metrics)} metrics")

    except subprocess.TimeoutExpired:
        print("Error: ncu --query-metrics timed out")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: ncu not found. Make sure Nsight Compute is in PATH")
        sys.exit(1)


class MetricSweeper:
    def __init__(self, output_base_dir: str = "metrics", injection_lib: str = None, test_binary: str = None, device_id: int = 0):
        self.script_dir = Path(__file__).parent.resolve()
        self.output_base_dir = Path(output_base_dir)
        self.device_id = device_id

        # Default paths relative to script location
        self.injection_lib = injection_lib or str(self.script_dir / "build" / "libpmsampling_injection.so")
        self.test_binary = test_binary or str(self.script_dir / "build" / "test_kernel")

        # Will be set after querying GPU
        self.gpu_name = None
        self.output_dir = None

        # Results tracking
        self.successful_metrics: List[str] = []
        self.failed_metrics: List[str] = []
        self.results: Dict[str, dict] = {}

    def query_metrics(self) -> List[Tuple[str, str, str]]:
        """Query all available metrics from ncu.

        Returns list of (metric_name, metric_type, metric_unit) tuples.
        """
        print("Querying available metrics from ncu...")

        try:
            result = subprocess.run(
                ["ncu", "--query-metrics"],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "TMPDIR": "/tmp"}
            )

            metrics = []
            # Check both stdout and stderr as ncu may output to either
            all_output = result.stdout + '\n' + result.stderr

            for line in all_output.split('\n'):
                # Skip header, separator, and special lines
                if not line.strip():
                    continue
                if line.startswith('-') or line.startswith('Metric Name'):
                    continue
                if line.startswith('Device') or line.startswith('*') or line.startswith('='):
                    continue

                # Split on whitespace - columns are separated by multiple spaces
                parts = line.split()
                if len(parts) >= 3:
                    metric_name = parts[0]
                    metric_type = parts[1]
                    metric_unit = parts[2]

                    # Only include valid metric names (start with letter, contain __)
                    if re.match(r'^[a-z][a-z0-9_]*__', metric_name):
                        metrics.append((metric_name, metric_type, metric_unit))

            print(f"Found {len(metrics)} metrics")
            return metrics

        except subprocess.TimeoutExpired:
            raise RuntimeError("ncu --query-metrics timed out")

    def save_metrics_list(self, metrics: List[Tuple[str, str, str]]):
        """Save the full metrics list to a file."""
        metrics_file = self.output_dir / "metrics.txt"

        with open(metrics_file, 'w') as f:
            f.write(f"# Metrics for {self.gpu_name}\n")
            f.write(f"# Total: {len(metrics)}\n")
            f.write("#" + "-" * 100 + "\n")
            f.write(f"# {'Metric Name':<70} {'Type':<15} {'Unit':<15}\n")
            f.write("#" + "-" * 100 + "\n")

            for name, mtype, unit in metrics:
                f.write(f"{name:<70} {mtype:<15} {unit:<15}\n")

        print(f"Saved metrics list to: {metrics_file}")

    def check_pm_sampling_support(self) -> bool:
        """Check if GPU supports PM sampling (compute capability >= 7.5).
        
        Returns True if supported, False otherwise.
        """
        try:
            # Use nvidia-smi to get compute capability for the selected device
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", f"-i={self.device_id}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print("Warning: Could not query GPU compute capability via nvidia-smi")
                return True
            
            compute_cap = result.stdout.strip()
            if not compute_cap:
                print("Warning: Could not parse compute capability from nvidia-smi")
                return True
            
            # Parse compute capability (format: "major.minor")
            parts = compute_cap.split('.')
            if len(parts) != 2:
                print(f"Warning: Unexpected compute capability format: {compute_cap}")
                return True
            
            major = int(parts[0])
            minor = int(parts[1])
            compute_value = major * 10 + minor
            
            # PM sampling requires compute capability >= 7.5
            if compute_value < 75:
                print(f"GPU compute capability {major}.{minor} does not support PM sampling (requires >= 7.5)")
                return False
            
            return True
        except Exception as e:
            print(f"Warning: Could not check GPU capability: {e}, assuming supported")
            return True

    def setup_environment(self, metric_name: str) -> Dict[str, str]:
        """Set up environment variables for CUPTI profiling."""
        env = os.environ.copy()

        # Set CUDA_VISIBLE_DEVICES to the selected device
        env['CUDA_VISIBLE_DEVICES'] = str(self.device_id)

        # Infer CUDA path from nvcc location, with fallbacks
        cuda_install_path = None
        try:
            nvcc_path = subprocess.check_output(['which', 'nvcc'], text=True).strip()
            if nvcc_path:
                # nvcc is typically at $CUDA_INSTALL_PATH/bin/nvcc, so go up two directories
                cuda_install_path = os.path.dirname(os.path.dirname(nvcc_path))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Fall back to CUDA_INSTALL_PATH env var if which nvcc failed
        if not cuda_install_path:
            cuda_install_path = env.get('CUDA_INSTALL_PATH', '/usr/local/cuda')
        
        cupti_lib_path = os.path.join(cuda_install_path, 'extras', 'CUPTI', 'lib64')
        cuda_lib_path = os.path.join(cuda_install_path, 'lib64')
        existing_ld_library_path = env.get('LD_LIBRARY_PATH', '')
        
        # Construct new LD_LIBRARY_PATH
        new_ld_library_path = f"{cupti_lib_path}:{cuda_lib_path}"
        if existing_ld_library_path:
            new_ld_library_path = f"{new_ld_library_path}:{existing_ld_library_path}"
        # env['LD_LIBRARY_PATH'] = new_ld_library_path

        # CUPTI injection settings
        env['CUDA_INJECTION64_PATH'] = self.injection_lib
        env['INJECTION_METRICS'] = metric_name
        env['INJECTION_KERNEL_COUNT'] = "10"
        env['PM_SAMPLING_HW_BUFFER_BYTES'] = "838860800"  # 800 MB
        env['PM_SAMPLING_INTERVAL_SYSCLK'] = "5000"
        env['PM_SAMPLING_MAX_SAMPLES'] = "16384"

        return env

    def test_metric(self, metric_name: str) -> Tuple[bool, str, float]:
        """Test a single metric with the injection library.

        Returns (success, error_message, duration).
        """
        env = self.setup_environment(metric_name)

        # Clean up any existing output files
        for f in Path('.').glob('output_*.csv'):
            f.unlink()

        start_time = time.time()
        try:
            result = subprocess.run(
                [self.test_binary],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
                cwd=str(self.output_dir)
            )
            duration = time.time() - start_time

            # Check for success - binary should return 0 and produce output
            if result.returncode == 0:
                # Check if any CSV was produced
                csv_files = list(self.output_dir.glob('output_*.csv'))
                if csv_files:
                    return True, "", duration
                else:
                    return True, "No CSV output", duration
            else:
                error = result.stderr[:500] if result.stderr else f"Return code: {result.returncode}"
                return False, error, duration

        except subprocess.TimeoutExpired:
            return False, "Timeout (>60s)", time.time() - start_time
        except Exception as e:
            return False, str(e), time.time() - start_time

    def run_sweep(self, metrics: List[Tuple[str, str, str]], metric_types: List[str] | None = None):
        """Run the metric sweep.

        Args:
            metrics: List of (name, type, unit) tuples
            metric_types: Filter to only these types (e.g., ['Counter']). None = all types.
        """
        # Filter metrics by type if specified
        if metric_types:
            metrics = [(n, t, u) for n, t, u in metrics if t in metric_types]

        # Add appropriate suffix based on metric type
        test_metrics = []
        for name, mtype, unit in metrics:
            if mtype == 'Counter':
                test_metrics.append(f"{name}.sum")
            elif mtype == 'Ratio':
                test_metrics.append(f"{name}.ratio")
            else:
                test_metrics.append(f"{name}.avg")

        total = len(test_metrics)
        print(f"\nTesting {total} metrics...")
        print("=" * 60)

        for i, metric in enumerate(test_metrics, 1):
            print(f"[{i}/{total}] {metric:<60} ", end="", flush=True)

            success, error, duration = self.test_metric(metric)

            if success:
                print(f"✓ ({duration:.1f}s)")
                self.successful_metrics.append(metric)
                self.results[metric] = {'status': 'success', 'duration': duration}
            else:
                print(f"✗ ({duration:.1f}s)")
                self.failed_metrics.append(metric)
                self.results[metric] = {'status': 'failed', 'error': error, 'duration': duration}

            # Brief pause between tests
            time.sleep(0.5)

    def save_results(self):
        """Save supported metrics to file."""
        # Supported metrics only (for easy use)
        supported_file = self.output_dir / "supported.txt"
        with open(supported_file, 'w') as f:
            for m in sorted(self.successful_metrics):
                f.write(f"{m}\n")

        print(f"\nResults saved to:")
        print(f"  {supported_file}")

    def run(self, metric_types: List[str] | None = None):
        """Main entry point."""
        # Check prerequisites
        if not os.path.exists(self.injection_lib):
            print(f"Error: Injection library not found: {self.injection_lib}")
            print("Please build first: cd build && cmake .. && make")
            sys.exit(1)

        if not os.path.exists(self.test_binary):
            print(f"Error: Test binary not found: {self.test_binary}")
            print("Please build first: cd build && cmake .. && make")
            sys.exit(1)

        # Check GPU compute capability support
        print("Checking GPU PM sampling support...")
        if not self.check_pm_sampling_support():
            print("Error: GPU does not support PM sampling (requires compute capability >= 7.5)")
            sys.exit(1)

        # Get GPU info
        print("Detecting GPU...")
        self.gpu_name, chip_name = get_gpu_info()
        print(f"GPU: {self.gpu_name} ({chip_name})")

        # Create output directory based on GPU name
        gpu_folder_name = self.gpu_name.replace(' ', '_').replace('-', '_')
        self.output_dir = self.output_base_dir / gpu_folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Query and save all metrics
        metrics = self.query_metrics()
        self.save_metrics_list(metrics)

        # Run the sweep
        self.run_sweep(metrics, metric_types)

        # Save results
        self.save_results()

        print("\n" + "=" * 60)
        print(f"Sweep complete!")
        print(f"Successful: {len(self.successful_metrics)}")
        print(f"Failed: {len(self.failed_metrics)}")
        print(f"Success rate: {len(self.successful_metrics) / max(1, len(self.successful_metrics) + len(self.failed_metrics)) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='CUPTI PM Sampling Metric Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Query and save pmsampling metrics (default)
  %(prog)s --output-dir ./my_dir    Save to specific directory
  %(prog)s --sweep                  Run full metric sweep with injection testing
  %(prog)s --sweep --counters-only  Sweep only Counter type metrics
"""
    )

    # Common arguments
    parser.add_argument('--output-dir', '-o', default='metrics',
                        help='Base output directory (default: metrics)')

    # Sweep mode arguments
    parser.add_argument('--sweep', action='store_true',
                        help='Run the full metric sweep with injection testing')
    parser.add_argument('-D', '--device', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    parser.add_argument('--injection-lib', default=None,
                        help='Path to libpmsampling_injection.so')
    parser.add_argument('--test-binary', default=None,
                        help='Path to test_kernel binary')
    parser.add_argument('--counters-only', action='store_true',
                        help='Only test Counter type metrics (sweep mode only)')

    args = parser.parse_args()

    if args.sweep:
        # Run the full metric sweep
        sweeper = MetricSweeper(
            output_base_dir=args.output_dir,
            injection_lib=args.injection_lib,
            test_binary=args.test_binary,
            device_id=args.device
        )

        metric_types = ['Counter'] if args.counters_only else None
        sweeper.run(metric_types)
    else:
        # Default: query pmsampling metrics and save to file
        query_pmsampling_metrics(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
