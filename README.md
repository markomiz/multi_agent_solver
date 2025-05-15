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

All dependencies can be automatically installed with the setup_dependencies.sh script 

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


## 📝 **Manual Build (Without Docker)**

If you prefer to build manually:

```bash
./scripts/setup_dependencies.sh
./scripts/build.sh
```

### **Run the Examples:**

```bash
./scripts/run.sh
```

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
========================================================================================
Outer Method                            Solver              Time (ms)           Cost                
========================================================================================
 Centralized                            CGD                 117.26              86.9948             
 Centralized                            OSQP                3685.16             75.9386             
 Centralized                            iLQR                2079.98             77.6986             
 Decentralized_Line_Search              CGD                 106.85              77.3723             
 Decentralized_Line_Search              OSQP                34.80               74.5437             
 Decentralized_Line_Search              iLQR                11.27               77.7030             
 Decentralized_Simple                   CGD                 138.99              77.7806             
 Decentralized_Simple                   OSQP                43.13               74.9514             
 Decentralized_Simple                   iLQR                10.17               78.1107             
 Decentralized_Trust_Region             CGD                 3.02                87.4026             
 Decentralized_Trust_Region             OSQP                9.96                76.3330             
 Decentralized_Trust_Region             iLQR                6.02                78.1107             
=======================================================================================
```


## License
Apache 2.0

## Project status
WIP  - still earlyl in development but working.
