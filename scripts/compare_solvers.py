#!/usr/bin/env python3
"""Utility to benchmark example executables across solver and strategy choices."""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
                    result = run_command(cmd, args.timeout, args.verbose)
                    if result.returncode != 0:
                        sys.stderr.write(
                            f"Error: '{example}' with solver={solver} strategy={strategy} exited with code {result.returncode}.\n"
                        )
                        if result.stderr:
                            sys.stderr.write(result.stderr + "\n")
                        if args.fail_fast:
                            return result.returncode or 1
                        continue
                    line = find_result_line(result.stdout)
                    if not line:
                        sys.stderr.write(
                            f"Error: could not parse output from '{example}' with solver={solver} strategy={strategy}.\n"
                        )
                        if args.fail_fast:
                            return 1
                        continue
                    data = parse_result_line(line)
                    data.setdefault("solver", solver)
                    data.setdefault("strategy", strategy)
                    results[example].append(data)
        else:
            for solver in args.solvers:
                cmd = [str(executable), f"--solver={solver}"]
                result = run_command(cmd, args.timeout, args.verbose)
                if result.returncode != 0:
                    sys.stderr.write(
                        f"Error: '{example}' with solver={solver} exited with code {result.returncode}.\n"
                    )
                    if result.stderr:
                        sys.stderr.write(result.stderr + "\n")
                    if args.fail_fast:
                        return result.returncode or 1
                    continue
                line = find_result_line(result.stdout)
                if not line:
                    sys.stderr.write(
                        f"Error: could not parse output from '{example}' with solver={solver}.\n"
                    )
                    if args.fail_fast:
                        return 1
                    continue
                data = parse_result_line(line)
                data.setdefault("solver", solver)
                results[example].append(data)

    for example in args.examples:
        rows = results.get(example, [])
        include_strategy = example in MULTI_AGENT_EXAMPLES
        print(f"\n=== {example} ===")
        print(format_table(rows, include_strategy))

    return 0


if __name__ == "__main__":
    sys.exit(main())
