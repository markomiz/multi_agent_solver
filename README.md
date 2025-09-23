# MultiAgentSolver

MultiAgentSolver is a high-performance C++ library designed to solve multi-agent optimization problems. The library includes solvers such as:

* **CGD (Constrained Gradient Descent)** 
* **iLQR (Iterative Linear Quadratic Regulator)** 
* **OSQP Solver** utilizing the osqp library

Additionally, it supports multi-agent coordination through Nash Equilibrium-based optimization and for comparison combining multiple agents into one big optimization problem.

## 🛠️ **Dependencies**

* **Eigen3** - Linear algebra
* **OSQP** - Optimization solver
* **OsqpEigen** - Eigen wrapper for OSQP
* **OpenMP** - Multi-threading support

All dependencies can be automatically installed with the `setup_dependencies.sh` script.

> **Tip:** On host machines you can customise how dependencies are installed:
>
> ```bash
> ./scripts/setup_dependencies.sh              # install system packages (if supported) and build third-party libraries
> ./scripts/setup_dependencies.sh --no-system-packages  # skip package manager operations (dependencies already installed)
> PREFIX="$HOME/.local" ./scripts/setup_dependencies.sh  # install OSQP/OsqpEigen into a custom prefix
> ```
>
> The script detects the operating system, chooses a supported package manager (`apt`, `brew`, `pacman`, `dnf`, or `yum`), and
> gracefully falls back if none is available. Third-party libraries are built into a user-writable prefix (`$HOME/.local` by
> default) and the `CMAKE_PREFIX_PATH` environment variable is configured so subsequent CMake invocations can discover them.
> The script also drops a reusable environment snippet at `$PREFIX/share/multi_agent_solver/environment.sh`; source it (or add
> it to your shell profile) to make the prefix available to manual CMake builds. When running in Docker/CI environments the
> defaults continue to work without additional flags.

---

## 📦 **Building with Docker**

The recommended way to build and run the project is using Docker. This ensures a clean environment and reproducible builds.

```bash
./scripts/run_docker.sh
```

This will:

* Clean up any existing Docker containers of the same name
* Rebuild the Docker image
* Run the container interactively

> **Note:** The Docker build context excludes local `build/` and `cmake-build-*` directories via `.dockerignore`, ensuring host
> build artifacts never sneak into container images.


## 📝 **Manual Build (Without Docker)**

If you prefer to build manually:

```bash
./scripts/setup_dependencies.sh
./scripts/build.sh
```

The build helper understands both `./scripts/build.sh Release` and `./scripts/build.sh --build-type Release`. It automatically
loads the generated dependency hints when `CMAKE_PREFIX_PATH` is unset so Docker builds and fresh shells continue to locate
OSQP/OsqpEigen without extra configuration. Use
`--clean` to start from a blank build directory; otherwise the script automatically removes the cached tree when
`CMAKE_SYSTEM_NAME` from a previous configuration does not match the current host (e.g., switching between local and
Docker builds).

### **Run the Examples:**

```bash
./scripts/run.sh
```

### **Utility scripts**

The repository includes Python helpers for benchmarking and visualising the example executables. Both scripts accept `--help` for the full list of options.

* **Compare solvers** – build the project (unless `--skip-build` is supplied) and benchmark multiple solver/strategy combinations:

  ```bash
  ./scripts/compare_solvers.py --build-type Release --examples multi_agent_lqr multi_agent_single_track \
      --solvers ilqr cgd --strategies centralized sequential --agents 8
  ```

  When the executables finish successfully, the script prints a compact table that lists the cost and runtime reported by each run.

* **Plot trajectories** – run a single example and plot the CSV-style trajectories it prints. Additional arguments after `--` are forwarded to the example executable:

  ```bash
  ./scripts/plot_example.py multi_agent_lqr -- --agents 4 --solver ilqr --strategy sequential
  ```

  By default the script opens an interactive Matplotlib window. Use `--output-dir plots --no-show` to save the generated figures as PNG files instead.

## 📂 **Project Structure**

```
├── CMakeLists.txt             # CMake build configuration
├── Dockerfile                 # Docker setup
├── include                    # Header files
│   └── MultiAgentSolver
├── src                        # Source files
├── examples                   # Example applications
├── scripts                    # Helper scripts
│   ├── build.sh
│   ├── run.sh
│   └── setup_dependencies.sh
├── cmake                      # CMake configuration files
│   └── MultiAgentSolverConfig.cmake.in
├── build                      # Build artifacts
└── README.md                  # Project documentation
```



## Results

Times and costs for different methods in the example code:
```
🚗 Single-Track Lane Following Test 🚗
---------------------------------------------
Solver              Cost           Time (ms)
     ---------------------------------------------
CGD                 24.0465        20.6444        
OSQP                30.1889        2.33275        
OSQP Collocation    23.9809        5.11993        
iLQR                24.4039        1.06887 
```

```

Multi-Agent Single Track Test

Method                                  Cost           Time (ms)      
----------------------------------------------------------------------
Centralized CGD                         7928.151       1214.919       
Centralized iLQR                        7928.501       135.472        
Centralized OSQP                        7929.011       285.711        
Centralized OSQP-collocation            7929.392       1071.582       
Nash Sequential CGD                     7928.153       26.612         
Nash Sequential iLQR                    7928.327       11.053         
Nash Sequential OSQP                    7928.384       38.514         
Nash Sequential OSQP-collocation        7928.158       2098.299       
Nash LineSearch CGD                     7928.153       27.597         
Nash LineSearch iLQR                    7928.327       13.807         
Nash LineSearch OSQP                    7928.384       40.643         
Nash LineSearch OSQP-collocation        7928.152       2010.733       
Nash TrustRegion CGD                    7928.153       34.767         
Nash TrustRegion iLQR                   7928.199       14.093         
Nash TrustRegion OSQP                   7928.417       46.087         
Nash TrustRegion OSQP-collocation       7928.152       1596.460 
```

## License
Apache 2.0

## Project status
WIP  - still earlyl in development but working.
