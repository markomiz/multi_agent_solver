# Contribution guide for MultiAgentSolver agents

## 1. Project overview & scope
MultiAgentSolver is a header-only C++20 library for multi-agent optimal control. It provides core solvers (conjugate gradient descent, iterative LQR, and OSQP-based variants) and coordination strategies inspired by Nash equilibria. Contributors primarily work within the `include/multi_agent_solver` directory, extending solver capabilities while keeping the public interface lightweight.

## 2. Dependencies & environment setup
- **Core tooling**: CMake ≥3.16, a C++20 compiler (GCC ≥11 or Clang ≥13), Eigen3, and OpenMP.
- **Optional**: OSQP and OsqpEigen for quadratic programming backends.
- **Helper script**: Run `scripts/install_dependencies.sh` to install prerequisites on Debian/Ubuntu systems via `apt-get` and build OSQP from source. Adjust manually for other environments.

## 3. Build, test, and example commands
- Configure & build: `scripts/build.sh`
- Run bundled examples: `scripts/run.sh`
- Docker workflow: Use the provided `Dockerfile` as a reproducible environment.
- CI parity: Mirror these scripts locally before submitting changes to avoid surprises in continuous integration.

## 4. Repository layout
- `include/multi_agent_solver`: Core header-only library, exported as an interface target.
- `examples/*.cpp`: Sample applications demonstrating solver usage.
- `scripts/*.py` & `scripts/*.sh`: Python and shell utilities for building, testing, and benchmarking.

## 5. C++ contribution guidelines
- Maintain the header-only design: keep headers self-contained, rely on `#pragma once`, and avoid introducing source files.
- Keep public APIs within the `mas` namespace. Document new classes and functions using the existing doc-comment style in solver headers.
- Favor `Eigen` types and utilities already used throughout the library to ensure consistency.

## 6. Formatting & linting
- Run `clang-format` using the repository’s Google-derived style (`.clang-format`).
- Match established conventions: 100 character line limit, Allman braces for functions, and vertically aligned assignments where already present.

## 7. Python scripting conventions
- Scripts should use type hints, `dataclasses`, and `argparse` for CLI surfaces.
- Follow PEP 8 naming, keep functions small and composable, and provide helpful `--help` descriptions for new arguments.

## 8. Testing & benchmarking expectations
- When altering solver logic, execute relevant examples or benchmarking utilities in `scripts/` to confirm performance and numerical stability.
- Update any documented tables or results if behavior changes materially, noting regressions or improvements in accompanying documentation or commit messages.

## 9. PR hygiene
- Keep commits focused and well-described. Update README or inline docs if user-facing behavior changes.
- Verify all scripts and tests pass before requesting review. Include a summary of verification steps in the PR description.
- Ensure the working tree is clean (`git status`) before committing and that CI-essential files (build scripts, Dockerfile) remain executable.
