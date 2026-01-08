#!/usr/bin/env python3
"""Run an example and generate an animated GIF of the trajectory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Import the plot_example script to reuse parsing logic
import plot_example

def animate_pendulum(
    trajectories: Dict[str, Dict[str, plot_example.Trajectory]],
    output_path: Path,
    show: bool
) -> None:
    data = trajectories.get("pendulum")
    if not data or "states" not in data:
        raise ValueError("Pendulum state trajectory not found.")

    states = data["states"]
    time = states.time
    # states: theta, omega. We only need theta (x0).
    theta = np.array(states.series["x0"])

    # Pendulum parameters (visual)
    L = 1.0

    x = L * np.sin(theta)
    y = L * np.cos(theta)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.5 * L, 1.5 * L)
    ax.set_ylim(-1.5 * L, 1.5 * L)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Pendulum Swing-up")

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'Time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [0, x[i]]
        thisy = [0, y[i]]
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % time[i])
        return line, time_text

    dt = time[1] - time[0] if len(time) > 1 else 0.05
    ani = animation.FuncAnimation(
        fig, animate, frames=len(time),
        init_func=init, interval=dt*1000, blit=True
    )

    if output_path:
        ani.save(output_path, writer='pillow', fps=int(1/dt))
        print(f"Saved animation to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def animate_single_track(
    trajectories: Dict[str, Dict[str, plot_example.Trajectory]],
    output_path: Path,
    show: bool
) -> None:
    # Identify agents
    agents = []
    for key in trajectories:
        if key.startswith("agent_"):
            agents.append(key)
    agents.sort()

    if not agents:
        raise ValueError("No agents found for single track animation.")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Track radius is 20.0 from C++ code
    track_radius = 20.0
    circle = plt.Circle((0, 0), track_radius, color='g', fill=False, linestyle='--', alpha=0.5, label='Track')
    ax.add_artist(circle)

    # Set limits based on track
    limit = track_radius * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Multi-Agent Single Track")

    lines = []
    points = []

    # Pre-process data
    agent_data = []
    max_frames = 0
    time = []

    colors = plt.cm.jet(np.linspace(0, 1, len(agents)))

    for i, agent in enumerate(agents):
        traj = trajectories[agent]["states"]
        x = np.array(traj.series["x0"])
        y = np.array(traj.series["x1"])
        agent_data.append((x, y))
        max_frames = max(max_frames, len(x))
        if len(traj.time) > len(time):
            time = traj.time

        line, = ax.plot([], [], '-', alpha=0.5, color=colors[i])
        point, = ax.plot([], [], 'o', color=colors[i], label=agent)
        lines.append(line)
        points.append(point)

    ax.legend(loc='upper right')

    time_template = 'Time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]

    def animate(i):
        for idx, (x, y) in enumerate(agent_data):
            if i < len(x):
                # Trace history
                lines[idx].set_data(x[:i+1], y[:i+1])
                # Current position
                points[idx].set_data([x[i]], [y[i]])

        if i < len(time):
            time_text.set_text(time_template % time[i])
        return lines + points + [time_text]

    dt = time[1] - time[0] if len(time) > 1 else 0.5
    ani = animation.FuncAnimation(
        fig, animate, frames=max_frames,
        init_func=init, interval=dt*1000, blit=True
    )

    if output_path:
        ani.save(output_path, writer='pillow', fps=int(1/dt))
        print(f"Saved animation to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def animate_rocket(
    trajectories: Dict[str, Dict[str, plot_example.Trajectory]],
    output_path: Path,
    show: bool
) -> None:
    data = trajectories.get("rocket")
    if not data or "states" not in data:
        raise ValueError("Rocket state trajectory not found.")

    states = data["states"]
    time = states.time
    # states: h, v, m. We use h (x0).
    h = np.array(states.series["x0"])

    fig, ax = plt.subplots(figsize=(6, 8))

    max_h = np.max(h)
    min_h = np.min(h)

    # Ensure reasonable y-limits even if trajectory is flat or negative
    if max_h == min_h:
        y_padding = 1.0
    else:
        y_padding = (max_h - min_h) * 0.1

    # Rocket starts at 0, so include 0 in the view
    y_min = min(0.0, min_h) - y_padding
    y_max = max(10.0, max_h) + y_padding # Ensure at least some height is shown

    ax.set_xlim(-1, 1)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Rocket Max Altitude")
    ax.set_ylabel("Altitude (m)")
    ax.set_xticks([]) # Remove x ticks as it is 1D motion

    # Adjust aspect if height range is large
    if (y_max - y_min) > 10:
         ax.set_aspect('auto')

    rocket_marker, = ax.plot([], [], 'r^', markersize=10, label='Rocket')
    trace, = ax.plot([], [], 'k--', alpha=0.3)

    time_template = 'Time = %.1fs\nAlt = %.1fm'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        rocket_marker.set_data([], [])
        trace.set_data([], [])
        time_text.set_text('')
        return rocket_marker, trace, time_text

    def animate(i):
        rocket_marker.set_data([0], [h[i]])
        trace.set_data(np.zeros(i+1), h[:i+1])
        time_text.set_text(time_template % (time[i], h[i]))
        return rocket_marker, trace, time_text

    dt = time[1] - time[0] if len(time) > 1 else 0.1
    ani = animation.FuncAnimation(
        fig, animate, frames=len(time),
        init_func=init, interval=dt*1000, blit=True
    )

    if output_path:
        ani.save(output_path, writer='pillow', fps=int(1/dt))
        print(f"Saved animation to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = plot_example.parse_arguments(argv)

    # Run the example using plot_example logic
    build_type = plot_example.canonical_build_type(args.build_type)

    example_args = list(args.example_args)
    if example_args and example_args[0] == "--":
        example_args = example_args[1:]

    if not args.skip_build and args.build_dir is None:
        try:
            plot_example.ensure_build(build_type, args.verbose)
        except (FileNotFoundError, RuntimeError) as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1

    if args.build_dir is not None:
        build_dir = args.build_dir.resolve()
    else:
        build_dir = (plot_example.REPO_ROOT / "build" / build_type.lower()).resolve()

    executable = build_dir / args.example
    if not executable.exists():
        sys.stderr.write(f"Error: executable '{executable}' not found.\n")
        return 1

    process = plot_example.run_example(executable, example_args, args.timeout, args.verbose)
    if process.returncode != 0:
        sys.stderr.write(
            f"Error: example exited with code {process.returncode}.\n"
        )
        if process.stderr:
            sys.stderr.write(process.stderr + "\n")
        return process.returncode or 1

    try:
        trajectories = plot_example.parse_trajectories(process.stdout)
    except ValueError as exc:
        sys.stderr.write(f"Error parsing example output: {exc}\n")
        return 1

    if not trajectories:
        print("No trajectory data found in example output.")
        return 0

    output_dir = plot_example.ensure_output_dir(args.output_dir)
    show = not args.no_show
    try:
        _, show = plot_example.load_pyplot(show)
    except RuntimeError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    # Dispatch to specific animator based on example name
    example_name = args.example

    output_path = None
    if output_dir:
        output_path = output_dir / f"{example_name}.gif"

    try:
        if "pendulum" in example_name:
            animate_pendulum(trajectories, output_path, show)
        elif "single_track" in example_name:
            animate_single_track(trajectories, output_path, show)
        elif "rocket" in example_name:
            animate_rocket(trajectories, output_path, show)
        else:
            print(f"No animation logic defined for example '{example_name}'.")
            # Fallback to static plot if no animation
            if not args.no_show: # Reuse the flag logic
                 # Avoid shadowing import
                 plot_example.plot_trajectories(trajectories, output_dir, show, plt)

    except Exception as e:
        sys.stderr.write(f"Error generating animation: {e}\n")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
