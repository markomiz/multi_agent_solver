#!/usr/bin/env python3
"""Utility to benchmark example executables across solver and strategy choices."""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _prepare_solution_agents(data: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    agents_raw = data.get("agents", [])
    if not isinstance(agents_raw, list):
        return {}

    prepared: Dict[Any, Dict[str, Any]] = {}
    for agent in agents_raw:
        if not isinstance(agent, dict):
            continue
        states = agent.get("states", [])
        if not isinstance(states, list) or not states:
            continue
        prepared[agent.get("id", 0)] = agent
    return prepared


def plot_example_solutions(
    example: str,
    solutions: List[Tuple[str, Optional[str], Dict[str, Any]]],
    plot_dir: Path,
) -> None:
    prepared: List[Tuple[str, Dict[Any, Dict[str, Any]]]] = []
    agent_order: List[Any] = []

    for solver, strategy, data in solutions:
        agent_map = _prepare_solution_agents(data)
        if not agent_map:
            continue
        label_parts = [solver]
        if strategy:
            label_parts.append(strategy)
        prepared.append((" / ".join(label_parts), agent_map))
        for identifier in agent_map.keys():
            if identifier not in agent_order:
                agent_order.append(identifier)

    if not prepared or not agent_order:
        strategy_names = sorted(
            {
                strategy
                for _, strategy, _ in solutions
                if strategy is not None
            }
        )
        detail = f" ({', '.join(strategy_names)})" if strategy_names else ""
        sys.stderr.write(
            f"Warning: no valid state data found for example '{example}'{detail}.\n"
        )
        return

    n_agents = len(agent_order)

    # Determine the dimensionality for each agent so we can create one column per
    # state component and hide unused subplots when some solutions omit an agent.
    agent_dims: Dict[Any, int] = {}
    for agent_id in agent_order:
        for _, agent_map in prepared:
            states = agent_map.get(agent_id, {}).get("states", [])
            if states and isinstance(states[0], list):
                agent_dims[agent_id] = len(states[0])
                break
        else:
            agent_dims[agent_id] = 0

    max_dims = max(agent_dims.values(), default=0)
    if max_dims == 0:
        sys.stderr.write(
            f"Warning: unable to determine state dimensionality for example '{example}'.\n"
        )
        return

    plt = get_pyplot()
    fig_width = max(4.0, 3.5 * max_dims)
    fig_height = max(2.5, 2.5 * n_agents)
    fig, axes = plt.subplots(
        n_agents,
        max_dims,
        squeeze=False,
        sharex=True,
        figsize=(fig_width, fig_height),
    )

    # Create combined colour and linestyle styles so each solution is visually distinct.
    color_cycle = plt.rcParams.get("axes.prop_cycle")
    if color_cycle is not None:
        colors = color_cycle.by_key().get("color", ["C0", "C1", "C2", "C3"])
    else:
        colors = ["C0", "C1", "C2", "C3"]
    line_styles = ["-", "--", "-.", ":"]
    style_pairs = []
    for color in colors:
        for linestyle in line_styles:
            style_pairs.append((color, linestyle))
    if not style_pairs:
        style_pairs = [("C0", "-")]

    legend_handles: List[Any] = []

    for sol_idx, (label, agent_map) in enumerate(prepared):
        color, linestyle = style_pairs[sol_idx % len(style_pairs)]

        for row, agent_id in enumerate(agent_order):
            dims = agent_dims.get(agent_id, 0)
            for dim in range(max_dims):
                ax = axes[row][dim]
                if dim >= dims:
                    ax.set_visible(False)
                    continue

                agent = agent_map.get(agent_id)
                if agent is None:
                    continue

                states = agent.get("states", [])
                if not states or not isinstance(states[0], list) or len(states[0]) <= dim:
                    continue

                dt = float(agent.get("dt", 1.0))
                times = [idx * dt for idx in range(len(states))]
                series = [step[dim] for step in states if len(step) > dim]
                if not series:
                    continue

                if row == 0 and dim == 0:
                    line, = ax.plot(
                        times[: len(series)],
                        series,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                    )
                    legend_handles.append(line)
                else:
                    ax.plot(
                        times[: len(series)],
                        series,
                        label=None,
                        color=color,
                        linestyle=linestyle,
                    )

                if row == n_agents - 1:
                    ax.set_xlabel("time [s]")
                if dim == 0:
                    ax.set_ylabel(f"agent {agent_id}")
                if row == 0:
                    ax.set_title(f"x{dim}")

    if legend_handles:
        labels = [line.get_label() for line in legend_handles]
        fig.legend(legend_handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    sup_title = f"{example} state trajectories"
    fig.suptitle(sup_title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    filename_parts = [example, "states"]
    filename = "_".join(sanitize_name(part) for part in filename_parts) + ".png"
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

    results: Dict[str, List[Dict[str, str]]] = {example: [] for example in args.examples}
    plot_requests: Dict[str, List[Tuple[str, Optional[str], Dict[str, Any]]]] = {
        example: [] for example in args.examples
    }

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
                            plot_requests[example].append(
                                (solver, strategy, solution_data)
                            )
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
                        plot_requests[example].append((solver, None, solution_data))

    if args.plot_states and plot_dir is not None:
        for example, solutions in plot_requests.items():
            if solutions:
                plot_example_solutions(example, solutions, plot_dir)

    for example in args.examples:
        rows = results.get(example, [])
        include_strategy = example in MULTI_AGENT_EXAMPLES
        print(f"\n=== {example} ===")
        print(format_table(rows, include_strategy))

    return 0


if __name__ == "__main__":
    sys.exit(main())
