#!/usr/bin/env python3
"""Run an example executable and plot its exported state/control trajectories."""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Trajectory:
    time: List[float]
    series: Dict[str, List[float]]


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "example",
        help="Name of the example executable to run (e.g. multi_agent_lqr).",
    )
    parser.add_argument(
        "example_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the example executable after '--'.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Path to the CMake build directory containing the example executables.",
    )
    parser.add_argument(
        "--build-type",
        default="Release",
        help=(
            "CMake build type passed to scripts/build.sh when building automatically "
            "(ignored if --build-dir is provided)."
        ),
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not invoke scripts/build.sh before running the example.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout (in seconds) for running the example executable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where generated plots will be saved (PNG).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the interactive Matplotlib window.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Echo the command executed and stderr from the example executable.",
    )
    return parser.parse_args(argv)


def canonical_build_type(value: str) -> str:
    mapping = {
        "debug": "Debug",
        "release": "Release",
        "relwithdebinfo": "RelWithDebInfo",
        "minsizerel": "MinSizeRel",
    }
    key = value.strip().lower()
    return mapping.get(key, value)


def ensure_build(build_type: str, verbose: bool) -> None:
    build_script = REPO_ROOT / "scripts" / "build.sh"
    if not build_script.exists():
        raise FileNotFoundError(f"Build script '{build_script}' not found.")
    cmd = [str(build_script), build_type]
    if verbose:
        print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Build failed with exit code "
            f"{result.returncode}. Run '{' '.join(cmd)}' manually for details."
        )


def run_example(
    executable: Path, example_args: Sequence[str], timeout: Optional[float], verbose: bool
) -> subprocess.CompletedProcess[str]:
    cmd = [str(executable), *example_args]
    if verbose:
        print("$", " ".join(cmd))
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if verbose and process.stderr:
        sys.stderr.write(process.stderr)
    return process


def parse_table(lines: Sequence[str], start_index: int) -> Tuple[Trajectory, int]:
    header = lines[start_index].strip()
    columns = [col.strip() for col in header.split(",") if col]
    if not columns or columns[0] != "time":
        raise ValueError(f"Expected CSV header starting with 'time', got '{header}'.")

    time_values: List[float] = []
    series: Dict[str, List[float]] = {col: [] for col in columns[1:]}

    index = start_index + 1
    while index < len(lines):
        row = lines[index].strip()
        if not row:
            break
        parts = [value.strip() for value in row.split(",")]
        if len(parts) != len(columns):
            raise ValueError(
                f"Row has {len(parts)} values but expected {len(columns)} entries: '{row}'."
            )
        try:
            time_values.append(float(parts[0]))
            for name, value in zip(columns[1:], parts[1:]):
                series[name].append(float(value))
        except ValueError as exc:
            raise ValueError(f"Failed to parse numeric value in row '{row}'.") from exc
        index += 1

    return Trajectory(time_values, series), index


def parse_trajectories(stdout: str) -> Dict[str, Dict[str, Trajectory]]:
    lines = stdout.splitlines()
    trajectories: Dict[str, Dict[str, Trajectory]] = {}
    index = 0

    while index < len(lines):
        label = lines[index].strip()
        if label.endswith("_states") or label.endswith("_controls"):
            try:
                base, kind = label.rsplit("_", 1)
            except ValueError:
                raise ValueError(f"Unexpected section label '{label}'.")
            if index + 1 >= len(lines):
                raise ValueError(f"Missing header for section '{label}'.")
            trajectory, index = parse_table(lines, index + 1)
            index += 1  # Skip the blank line separator.
            data = trajectories.setdefault(base, {})
            data[kind] = trajectory
        else:
            index += 1
    return trajectories


def ensure_output_dir(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def plot_trajectories(
    trajectories: Dict[str, Dict[str, Trajectory]],
    output_dir: Optional[Path],
    show: bool,
    plt: Any,
) -> None:
    for label in sorted(trajectories):
        sections = []
        data = trajectories[label]
        if "states" in data:
            sections.append(("states", data["states"]))
        if "controls" in data:
            sections.append(("controls", data["controls"]))
        if not sections:
            continue

        fig, axes = plt.subplots(
            len(sections),
            1,
            sharex=True,
            figsize=(10, 4 * len(sections)),
        )
        try:
            axes_array = list(axes)
        except TypeError:
            axes_array = [axes]

        for ax, (kind, traj) in zip(axes_array, sections):
            for name, values in traj.series.items():
                ax.plot(traj.time, values, label=name)
            ax.set_ylabel(kind.capitalize())
            ax.set_title(f"{label} {kind}")
            ax.grid(True, linestyle="--", alpha=0.5)
            if traj.series:
                ax.legend()
        axes_array[-1].set_xlabel("time")
        fig.tight_layout()

        if output_dir is not None:
            filename = output_dir / f"{label}.png"
            fig.savefig(filename)
            print(f"Saved plot to {filename}")

    if show:
        plt.show()
    else:
        plt.close("all")


def load_pyplot(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - import error surface to user
        message = "Failed to import matplotlib.pyplot. Install a GUI backend or run with --no-show."
        raise RuntimeError(message) from exc

    return plt


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)
    build_type = canonical_build_type(args.build_type)

    example_args = list(args.example_args)
    if example_args and example_args[0] == "--":
        example_args = example_args[1:]

    if not args.skip_build and args.build_dir is None:
        try:
            ensure_build(build_type, args.verbose)
        except (FileNotFoundError, RuntimeError) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1

    if args.build_dir is not None:
        build_dir = args.build_dir.resolve()
    else:
        build_dir = (REPO_ROOT / "build" / build_type.lower()).resolve()

    executable = build_dir / args.example
    if not executable.exists():
        sys.stderr.write(f"Error: executable '{executable}' not found.\n")
        return 1

    process = run_example(executable, example_args, args.timeout, args.verbose)
    if process.returncode != 0:
        sys.stderr.write(
            f"Error: example exited with code {process.returncode}.\n"
        )
        if process.stderr:
            sys.stderr.write(process.stderr + "\n")
        return process.returncode or 1

    try:
        trajectories = parse_trajectories(process.stdout)
    except ValueError as exc:
        sys.stderr.write(f"Error parsing example output: {exc}\n")
        return 1

    if not trajectories:
        print("No trajectory data found in example output.")
        return 0

    output_dir = ensure_output_dir(args.output_dir)
    show = not args.no_show
    try:
        plt = load_pyplot(show)
    except RuntimeError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    plot_trajectories(trajectories, output_dir, show, plt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
