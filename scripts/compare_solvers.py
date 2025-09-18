#!/usr/bin/env python3
"""Utility to benchmark example executables across solver and strategy choices."""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]


MULTI_AGENT_EXAMPLES = {
    "multi_agent_lqr",
    "multi_agent_single_track",
}
SINGLE_AGENT_EXAMPLES = {
    "single_track_ocp",
    "pendulum_swing_up",
}

ALL_EXAMPLES = tuple(sorted(MULTI_AGENT_EXAMPLES | SINGLE_AGENT_EXAMPLES))


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Path to the CMake build directory that contains the example executables.",
    )
    parser.add_argument(
        "--build-type",
        default="Release",
        help=(
            "CMake build type to pass to scripts/build.sh when building automatically "
            "(ignored if --build-dir is supplied)."
        ),
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not invoke scripts/build.sh before running the comparisons.",
    )
    parser.add_argument(
        "--examples",
        nargs="*",
        default=list(ALL_EXAMPLES),
        choices=ALL_EXAMPLES,
        help="Subset of examples to benchmark.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["ilqr", "cgd", "osqp", "osqpcollocation"],
        help="Solvers to test. Names are passed directly to the executables.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["centralized", "sequential", "linesearch", "trustregion"],
        help="Strategies to test for multi-agent examples.",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=10,
        help="Number of agents for multi-agent examples.",
    )
    parser.add_argument(
        "--max-outer",
        dest="max_outer",
        type=int,
        default=10,
        help="Maximum outer iterations for Nash strategies.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout (in seconds) for each executable invocation.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first failing executable invocation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print commands before executing them and echo their stderr output.",
    )
    parser.add_argument(
        "--plot-states",
        action="store_true",
        help="Generate matplotlib plots of state trajectories for successful runs.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to store generated plots (defaults to ./plots when --plot-states is used).",
    )
    return parser.parse_args(argv)


def run_command(cmd: List[str], timeout: Optional[float], verbose: bool) -> CommandResult:
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
    return CommandResult(process.stdout, process.stderr, process.returncode)


def find_result_line(output: str) -> Optional[str]:
    for line in reversed(output.splitlines()):
        if "cost=" in line and "time_ms=" in line:
            return line.strip()
    return None


def parse_result_line(line: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        data[key] = value
    return data


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
            f"Build failed with exit code {result.returncode}. Run '{' '.join(cmd)}' manually for details."
        )


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


_PYPLOT = None


def get_pyplot():  # type: ignore[override]
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        _PYPLOT = plt
    return _PYPLOT


def load_solution_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        sys.stderr.write(f"Warning: solution dump '{path}' was not created.\n")
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Warning: failed to parse JSON from '{path}': {exc}\n")
    return None


def plot_state_trajectories(data: Dict[str, Any], example: str, plot_dir: Path) -> None:
    agents = data.get("agents", [])
    if not isinstance(agents, list):
        sys.stderr.write("Warning: malformed solution JSON (missing agents list).\n")
        return

    solver = str(data.get("solver", "unknown"))
    strategy = data.get("strategy")
    if strategy is not None:
        strategy = str(strategy)

    plt = get_pyplot()
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        states = agent.get("states", [])
        if not states:
            continue
        dt = float(agent.get("dt", 1.0))
        times = [idx * dt for idx in range(len(states))]
        dims = len(states[0]) if states and isinstance(states[0], list) else 0
        if dims == 0:
            continue

        fig, ax = plt.subplots()
        for dim in range(dims):
            series = [step[dim] for step in states if len(step) > dim]
            if not series:
                continue
            ax.plot(times[: len(series)], series, label=f"x{dim}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("state")

        title_parts = [example, solver]
        if strategy:
            title_parts.append(strategy)
        title_parts.append(f"agent {agent.get('id', 0)}")
        ax.set_title(" / ".join(title_parts))
        if dims > 1:
            ax.legend()
        fig.tight_layout()

        filename_parts = [example, solver]
        if strategy:
            filename_parts.append(strategy)
        filename_parts.append(f"agent{agent.get('id', 0)}")
        filename = "_".join(sanitize_name(part) for part in filename_parts) + "_states.png"
        save_path = plot_dir / filename
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved state plot to {save_path}")


def format_table(rows: List[Dict[str, str]], include_strategy: bool) -> str:
    if not rows:
        return "(no successful runs)"

    def fmt_float(value: str, precision: int = 2) -> str:
        try:
            return f"{float(value):.{precision}f}"
        except ValueError:
            return value

    headers = ["solver"]
    if include_strategy:
        headers.append("strategy")
    headers.extend(["cost", "time_ms"])

    display_rows: List[List[str]] = []
    for row in rows:
        solver = row.get("solver", "")
        cost = fmt_float(row.get("cost", ""))
        time_ms = fmt_float(row.get("time_ms", ""))
        values = [solver]
        if include_strategy:
            values.append(row.get("strategy", ""))
        values.extend([cost, time_ms])
        display_rows.append(values)

    widths = [len(header) for header in headers]
    for row in display_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    lines = [
        "    ".join(header.ljust(widths[idx])
                    for idx, header in enumerate(headers)),
        "-" * (sum(widths) + (len(widths) * 4)),
    ]
    for row in display_rows:
        lines.append("    ".join(value.ljust(widths[idx])
                     for idx, value in enumerate(row)))
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)
    build_type = canonical_build_type(args.build_type)

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

    plot_dir: Optional[Path] = None
    if args.plot_states:
        plot_dir = (args.plot_dir or Path("plots")).resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, List[Dict[str, str]]] = {
        example: [] for example in args.examples}

    for example in args.examples:
        executable = build_dir / example
        if not executable.exists():
            sys.stderr.write(
                f"Warning: executable '{executable}' not found. Skipping {example}.\n")
            continue

        if example in MULTI_AGENT_EXAMPLES:
            for solver in args.solvers:
                for strategy in args.strategies:
                    cmd = [
                        str(executable),
                        f"--solver={solver}",
                        f"--strategy={strategy}",
                        f"--agents={args.agents}",
                        f"--max-outer={args.max_outer}",
                    ]
                    dump_path: Optional[Path] = None
                    if args.plot_states:
                        tmp = tempfile.NamedTemporaryFile(prefix=f"{example}_", suffix=".json", delete=False)
                        dump_path = Path(tmp.name)
                        tmp.close()
                        cmd.append(f"--dump-json={dump_path}")
                    result = run_command(cmd, args.timeout, args.verbose)
                    if result.returncode != 0:
                        sys.stderr.write(
                            f"Error: '{example}' with solver={solver} strategy={strategy} exited with code {result.returncode}.\n"
                        )
                        if result.stderr:
                            sys.stderr.write(result.stderr + "\n")
                        if dump_path is not None and dump_path.exists():
                            dump_path.unlink()
                        if args.fail_fast:
                            return result.returncode or 1
                        continue
                    line = find_result_line(result.stdout)
                    if not line:
                        sys.stderr.write(
                            f"Error: could not parse output from '{example}' with solver={solver} strategy={strategy}.\n"
                        )
                        if dump_path is not None and dump_path.exists():
                            dump_path.unlink()
                        if args.fail_fast:
                            return 1
                        continue
                    data = parse_result_line(line)
                    data.setdefault("solver", solver)
                    data.setdefault("strategy", strategy)
                    results[example].append(data)
                    if args.plot_states and dump_path is not None:
                        solution_data = load_solution_json(dump_path)
                        if dump_path.exists():
                            dump_path.unlink()
                        if solution_data and plot_dir is not None:
                            plot_state_trajectories(solution_data, example, plot_dir)
        else:
            for solver in args.solvers:
                cmd = [str(executable), f"--solver={solver}"]
                dump_path: Optional[Path] = None
                if args.plot_states:
                    tmp = tempfile.NamedTemporaryFile(prefix=f"{example}_", suffix=".json", delete=False)
                    dump_path = Path(tmp.name)
                    tmp.close()
                    cmd.append(f"--dump-json={dump_path}")
                result = run_command(cmd, args.timeout, args.verbose)
                if result.returncode != 0:
                    sys.stderr.write(
                        f"Error: '{example}' with solver={solver} exited with code {result.returncode}.\n"
                    )
                    if result.stderr:
                        sys.stderr.write(result.stderr + "\n")
                    if dump_path is not None and dump_path.exists():
                        dump_path.unlink()
                    if args.fail_fast:
                        return result.returncode or 1
                    continue
                line = find_result_line(result.stdout)
                if not line:
                    sys.stderr.write(
                        f"Error: could not parse output from '{example}' with solver={solver}.\n"
                    )
                    if dump_path is not None and dump_path.exists():
                        dump_path.unlink()
                    if args.fail_fast:
                        return 1
                    continue
                data = parse_result_line(line)
                data.setdefault("solver", solver)
                results[example].append(data)
                if args.plot_states and dump_path is not None:
                    solution_data = load_solution_json(dump_path)
                    if dump_path.exists():
                        dump_path.unlink()
                    if solution_data and plot_dir is not None:
                        plot_state_trajectories(solution_data, example, plot_dir)

    for example in args.examples:
        rows = results.get(example, [])
        include_strategy = example in MULTI_AGENT_EXAMPLES
        print(f"\n=== {example} ===")
        print(format_table(rows, include_strategy))

    return 0


if __name__ == "__main__":
    sys.exit(main())
